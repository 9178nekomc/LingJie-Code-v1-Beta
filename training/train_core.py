# -*- coding: utf-8 -*-
"""
LingJie-Code-v1-Pro  —  训练核心
  - BF16 自动混合精度 (torch.amp)
  - 梯度累积 -> 全局 batch 256
  - torch.nn.utils.clip_grad_norm_(max_norm=1.0)（必加，防爆）
  - total_loss = lm + 0.1*intent + 0.1*parser + 0.05*critic + 0.05*template + 0.1*verb
  - 辅助 Head CrossEntropy ignore_index=-100
  - 每 100 步打印诊断
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "tokenizer"))

from tokenizer import Tokenizer, RESP_ID
from lingjie_code import LingJieCodeModel

TASKS = ["intent", "parse", "verb", "tmpl", "gen", "critic"]


class PackedDataset:
    """从 dataset.npz 读取，按任务分桶采样"""

    def __init__(self, path):
        z = np.load(path)
        self.ids = z["ids"]
        self.parser = z["parser"]
        self.lens = z["lens"]
        self.task = z["task"]
        self.resp_pos = z["resp_pos"]
        self.intent = z["intent"]
        self.verb = z["verb"]
        self.tmpl = z["tmpl"]
        self.critic = z["critic"]
        self.offsets = np.zeros(len(self.lens) + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.lens)
        self.task_index = {t: np.where(self.task == t)[0] for t in range(6)}

    def __len__(self):
        return len(self.lens)

    def get(self, i):
        s, e = self.offsets[i], self.offsets[i + 1]
        return {
            "ids": self.ids[s:e],
            "parser": self.parser[s:e],
            "task": int(self.task[i]),
            "resp_pos": int(self.resp_pos[i]),
            "intent": int(self.intent[i]),
            "verb": int(self.verb[i]),
            "tmpl": int(self.tmpl[i]),
            "critic": int(self.critic[i]),
        }


def collate(samples, pad_id=0):
    bsz = len(samples)
    maxlen = max(len(s["ids"]) for s in samples)
    ids = torch.full((bsz, maxlen), pad_id, dtype=torch.long)
    parser = torch.full((bsz, maxlen), -100, dtype=torch.long)
    resp_pos = torch.zeros(bsz, dtype=torch.long)
    lens = torch.zeros(bsz, dtype=torch.long)
    for i, s in enumerate(samples):
        L = len(s["ids"])
        ids[i, :L] = torch.from_numpy(s["ids"].astype(np.int64))
        parser[i, :L] = torch.from_numpy(s["parser"].astype(np.int64))
        resp_pos[i] = s["resp_pos"]
        lens[i] = L
    def aux(name):
        v = [s[name] for s in samples]
        return torch.tensor(v, dtype=torch.long)
    return {"ids": ids, "parser": parser, "resp_pos": resp_pos,
            "lens": lens, "intent": aux("intent"), "verb": aux("verb"),
            "tmpl": aux("tmpl"), "critic": aux("critic")}


def compute_losses(model, batch, device):
    ids = batch["ids"].to(device, non_blocking=True)
    bsz, seq = ids.shape
    out = model(ids, pool_idx=batch["resp_pos"].to(device))
    lm_logits = out["lm_logits"]

    # LM loss：只在 <resp> 之后的 target 上算（预测 label/code/eos）
    targets = ids[:, 1:].clone()
    pos = torch.arange(seq - 1, device=device).unsqueeze(0)
    mask = pos >= batch["resp_pos"].to(device).unsqueeze(1)   # [b, seq-1]
    targets[~mask] = -100
    lm_loss = F.cross_entropy(lm_logits[:, :-1].reshape(-1, lm_logits.size(-1)),
                              targets.reshape(-1), ignore_index=-100)

    def seq_ce(logits, labels):
        valid = labels != -100
        if valid.sum() == 0:
            return lm_logits.new_zeros(())
        return F.cross_entropy(logits[valid], labels[valid])

    intent_loss = seq_ce(out["intent"], batch["intent"].to(device))
    verb_loss = seq_ce(out["verb"], batch["verb"].to(device))
    tmpl_loss = seq_ce(out["template"], batch["tmpl"].to(device))
    critic_loss = seq_ce(out["critic"], batch["critic"].to(device))
    parser_valid = (batch["parser"] != -100)
    if parser_valid.any():
        parser_loss = F.cross_entropy(
            out["parser"].reshape(-1, 4),
            batch["parser"].to(device).reshape(-1), ignore_index=-100)
    else:
        parser_loss = lm_logits.new_zeros(())

    total = (lm_loss + 0.1 * intent_loss + 0.1 * parser_loss +
             0.05 * critic_loss + 0.05 * tmpl_loss + 0.1 * verb_loss)
    return total, {
        "total": total, "lm": lm_loss, "intent": intent_loss,
        "parser": parser_loss, "critic": critic_loss,
        "template": tmpl_loss, "verb": verb_loss,
    }


def make_batch_sampler(ds, batch_size, seed, task_mix=None):
    """按任务分桶采样，任务间按 mix 比例轮转"""
    rng = np.random.RandomState(seed)
    counters = {t: 0 for t in range(6)}
    orders = {}
    for t, idxs in ds.task_index.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        orders[t] = idxs
    if task_mix is None:
        task_mix = [0.10, 0.19, 0.15, 0.12, 0.33, 0.11]  # 近似数据占比
    while True:
        t = rng.choice(6, p=task_mix)
        idxs = orders[t]
        take = []
        for _ in range(batch_size):
            if counters[t] >= len(idxs):
                rng.shuffle(idxs)
                counters[t] = 0
            take.append(idxs[counters[t]])
            counters[t] += 1
        yield [ds.get(i) for i in take]


def train(model, ds, out_dir, lr=3e-4, epochs=1.0, micro_bs=32,
          accum=8, warmup=300, seed=42, log_every=100, max_steps=None,
          task_mix=None, save_every=1000, resume_opt=None):
    device = "cuda"
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sampler = make_batch_sampler(ds, micro_bs, seed, task_mix)
    steps_per_epoch = (len(ds) // (micro_bs * accum)) + 1
    total_steps = int(steps_per_epoch * epochs)
    if max_steps:
        total_steps = min(total_steps, max_steps)

    def lr_at(step):
        if step < warmup:
            return lr * step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return lr * (0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")
    step = 0
    t0 = time.time()
    tokens_seen = 0
    print("total_steps:", total_steps, "steps/epoch:", steps_per_epoch)
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        agg = {}
        for micro in range(accum):
            batch = collate(next(sampler))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                total, parts = compute_losses(model, batch, device)
            total = total / accum
            total.backward()
            tokens_seen += int(batch["lens"].sum())
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v.item() / accum
        # 梯度裁剪（必加）：防小模型联合训练梯度爆炸
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        cur_lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.step()
        step += 1

        if step % log_every == 0 or step == 1:
            rec = {"step": step, "lr": round(cur_lr, 8),
                   "gnorm": round(float(gnorm), 4),
                   "tok_s": int(tokens_seen / (time.time() - t0)),
                   **{k: round(v, 4) for k, v in agg.items()}}
            print("Step %d | Total: %.4f | LM: %.4f | Intent: %.4f | Parser: %.4f"
                  " | Verb: %.4f | Tmpl: %.4f | Critic: %.4f | gnorm: %.3f | lr: %.2e"
                  % (step, agg["total"], agg["lm"], agg["intent"],
                     agg["parser"], agg["verb"], agg["template"],
                     agg["critic"], float(gnorm), cur_lr), flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        if step % save_every == 0 or step == total_steps:
            torch.save({"model": model.state_dict(),
                        "config": model.config},
                       os.path.join(out_dir, "ckpt_latest.pt"))
    torch.save({"model": model.state_dict(), "config": model.config},
               os.path.join(out_dir, "ckpt_final.pt"))
    print("saved final ->", out_dir, "elapsed:", round(time.time() - t0, 1), "s")
