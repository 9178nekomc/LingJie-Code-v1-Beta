# -*- coding: utf-8 -*-
"""SFT：预训练后以 1e-5 低学习率精调，强化 元数据->代码 与决策链质量。
任务配比向 gen 倾斜。"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from train_core import PackedDataset, train
from lingjie_code import LingJieCodeModel
from tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro-bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "sft"))
    args = ap.parse_args()

    tk = Tokenizer.load(os.path.join(ROOT, "tokenizer", "tokenizer.json"))
    ds = PackedDataset(os.path.join(ROOT, "data", "dataset.npz"))

    ck = torch_load(args.ckpt)
    model = LingJieCodeModel(ck["config"], pad_token_id=0, resp_token_id=4)
    model.load_state_dict(ck["model"])
    print("loaded ckpt, params: %.1fM" % (model.num_parameters() / 1e6))

    # SFT 任务配比：gen 50% / parse 12% / verb 8% / intent 10% / critic 12% / tmpl 8%
    train(model, ds, args.out, lr=args.lr, epochs=args.epochs,
          micro_bs=args.micro_bs, accum=args.accum, warmup=100,
          task_mix=[0.10, 0.12, 0.08, 0.08, 0.50, 0.12])


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    main()
