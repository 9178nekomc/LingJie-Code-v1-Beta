# -*- coding: utf-8 -*-
"""诊断：sft2 模型 + 当前 tokenizer 是否自洽"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "model")

import torch
import torch.nn.functional as F

from tokenizer import Tokenizer
from lingjie_code import LingJieCodeModel

tk = Tokenizer.load("tokenizer/tokenizer.json")
ck = torch.load("runs/sft2/ckpt_final.pt", map_location="cpu",
                weights_only=False)
model = LingJieCodeModel(ck["config"], pad_token_id=0, resp_token_id=4)
model.load_state_dict(ck["model"])
model.cuda().eval()

# 1) 训练分布内的 teacher-forcing loss
seq = '<task_parse>输入一个数，如果大于10打印big，否则打印small<resp>{"action": "input", "data": ["x"], "condition": "x < 10", "body": ["if x < 10: print(\\"big\\")", "else: print(\\"small\\")"]}'
ids = torch.tensor([tk.encode(seq) + [2]], device="cuda")
with torch.no_grad():
    out = model(ids)
logits = out["lm_logits"][0, :-1]
targets = ids[0, 1:]
loss = F.cross_entropy(logits, targets)
print("teacher-forcing loss on trained-style seq:", loss.item())

# 2) 全序列贪心（无 cache）
cur = ids[:, :ids.shape[1] - 60]
for _ in range(60):
    with torch.no_grad():
        o = model(cur, with_heads=False)
    nt = o["lm_logits"][0, -1].argmax().item()
    cur = torch.cat([cur, torch.tensor([[nt]], device="cuda")], dim=1)
    if nt == 2:
        break
print("full-forward greedy:", repr(tk.decode(cur[0, ids.shape[1] - 60:].tolist())))

# 3) generate（DualState Cache）
gen, _ = model.generate(cur, max_new_tokens=60, temperature=0.0, use_cache=True)
print("cached generate:", repr(tk.decode(gen[0, cur.shape[1]:].tolist())))
