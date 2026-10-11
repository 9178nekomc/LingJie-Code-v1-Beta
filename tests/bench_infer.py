# -*- coding: utf-8 -*-
"""推理测速：JAM-DualState Cache 增量解码 vs 全序列重算"""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "model")

import torch

from tokenizer import Tokenizer
from lingjie_code import LingJieCodeModel

tk = Tokenizer.load("tokenizer/tokenizer.json")
ck = torch.load("runs/sft/ckpt_final.pt", map_location="cpu",
                weights_only=False)
model = LingJieCodeModel(ck["config"], pad_token_id=0, resp_token_id=4)
model.load_state_dict(ck["model"])
model.cuda().eval()

ids = torch.tensor([[tk.token2id["<task_gen>"]] +
                    tk.encode('{"action": "print", "data": ["Hello World"]}') +
                    [tk.token2id["<resp>"]]], device="cuda")
N = 80


def timed(use_cache):
    # warmup
    model.generate(ids, max_new_tokens=20, temperature=0.0, use_cache=use_cache)
    torch.cuda.synchronize()
    t0 = time.time()
    n_tok = 0
    for _ in range(3):
        gen, _ = model.generate(ids, max_new_tokens=N, temperature=0.0,
                                use_cache=use_cache)
        n_tok += N
    torch.cuda.synchronize()
    return n_tok / (time.time() - t0)


fast = timed(True)
slow = timed(False)
print("with JAM-DualState Cache: %.1f tok/s" % fast)
print("without cache (full recompute): %.1f tok/s" % slow)
print("speedup: %.2fx" % (fast / slow))
