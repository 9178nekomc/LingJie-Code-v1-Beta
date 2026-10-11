# -*- coding: utf-8 -*-
"""吞吐基准：micro_bs=32 x accum=8 时的 tok/s 与显存"""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "training")
sys.path.insert(0, "model")

import torch

from train_core import PackedDataset, collate, make_batch_sampler, compute_losses
from lingjie_code import LingJieCodeModel

ds = PackedDataset("data/dataset.npz")
model = LingJieCodeModel({"vocab_size": 8192, "n_layers": 52},
                         pad_token_id=0, resp_token_id=4).cuda()
sampler = make_batch_sampler(ds, 32, 42)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

t0 = time.time()
tok = 0
for step in range(12):
    opt.zero_grad(set_to_none=True)
    for m in range(8):
        b = collate(next(sampler))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            total, parts = compute_losses(model, b, "cuda")
        (total / 8).backward()
        tok += int(b["lens"].sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step == 1:
        torch.cuda.synchronize()
        t0 = time.time()
        tok = 0
torch.cuda.synchronize()
dt = time.time() - t0
print("measured steps: 10 | tok/s: %.0f | mem: %.1f GB"
      % (tok / dt, torch.cuda.max_memory_allocated() / 1e9))
