# -*- coding: utf-8 -*-
"""预训练入口。--smoke 用小样本/小步数验证训练管线与 Loss 稳定性。"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from train_core import PackedDataset, train
from lingjie_code import LingJieCodeModel
from tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=float, default=1.5)
    ap.add_argument("--micro-bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--layers", type=int, default=52)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "pretrain"))
    args = ap.parse_args()

    tk = Tokenizer.load(os.path.join(ROOT, "tokenizer", "tokenizer.json"))
    ds = PackedDataset(os.path.join(ROOT, "data", "dataset.npz"))
    print("dataset:", len(ds), "samples")

    model = LingJieCodeModel(
        {"vocab_size": tk.vocab_size, "n_layers": args.layers},
        pad_token_id=0, resp_token_id=4)
    n = model.num_parameters()
    print("params: %.1fM" % (n / 1e6))
    assert 150e6 <= n <= 700e6, "param out of range"

    if args.smoke:
        # 冒烟：只取 4000 条，跑 40 步，验证 Loss 收敛方向与显存
        sub = {t: idx[:700] for t, idx in ds.task_index.items()}
        for t, idx in sub.items():
            ds.task_index[t] = idx
        train(model, ds, os.path.join(ROOT, "runs", "smoke"), lr=3e-4,
              epochs=1, micro_bs=2, accum=1, warmup=10, log_every=5,
              max_steps=40, save_every=1000)
    else:
        train(model, ds, args.out, lr=args.lr, epochs=args.epochs,
              micro_bs=args.micro_bs, accum=args.accum)


if __name__ == "__main__":
    main()
