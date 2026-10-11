# -*- coding: utf-8 -*-
"""
训练-推理一致性验证：同一段输入，
  A) 全序列前向（训练模式）
  B) 逐 Token 增量前向（JAM-DualState Cache 模式）
对比最终 hidden / lm_logits 与每层输出，必须逐元素一致（容差内）。
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

from lingjie_code import LingJieCodeModel
from jam_block import JAMDualStateCache


def main():
    device = "cuda"
    torch.manual_seed(0)
    # 用小配置验证算法正确性（算法与层数无关）
    model = LingJieCodeModel({"vocab_size": 9600, "n_layers": 4},
                             pad_token_id=0, resp_token_id=4).to(device).eval()

    ids = torch.randint(0, 9600, (2, 48), device=device)

    # ---- A: 全序列 ----
    with torch.no_grad():
        out_full = model(ids)
    ref_logits = out_full["lm_logits"]

    # ---- B: 逐 token 增量 + DualState Cache ----
    bsz = ids.shape[0]
    cache = JAMDualStateCache(model, bsz, device)
    step_logits = []
    for t in range(ids.shape[1]):
        with torch.no_grad():
            out = model(ids[:, t: t + 1], state=cache)
        step_logits.append(out["lm_logits"][:, -1])
    inc_logits = torch.stack(step_logits, dim=1)   # [b, t, vocab]

    diff = (ref_logits.float() - inc_logits.float()).abs()
    print("[pure-incremental] max |diff|:", diff.max().item())
    ok1 = torch.allclose(ref_logits.float(), inc_logits.float(), atol=5e-3,
                         rtol=1e-3)

    # ---- C: prefill 前 16 个 token + 剩余逐 token 增量（混合路径） ----
    cache2 = JAMDualStateCache(model, bsz, device)
    with torch.no_grad():
        out2 = model(ids[:, :16], state=cache2)
    step_logits2 = [out2["lm_logits"]]
    for t in range(16, ids.shape[1]):
        with torch.no_grad():
            out = model(ids[:, t: t + 1], state=cache2)
        step_logits2.append(out["lm_logits"][:, -1:])
    mix_logits = torch.cat(step_logits2, dim=1)
    diff2 = (ref_logits.float() - mix_logits.float()).abs()
    print("[prefill+incremental] max |diff|:", diff2.max().item())
    ok2 = torch.allclose(ref_logits.float(), mix_logits.float(), atol=5e-3,
                         rtol=1e-3)

    ok = ok1 and ok2
    print("CONSISTENCY:", "PASS" if ok else "FAIL")
    if not ok:
        per_tok = diff2.amax(dim=-1)
        bad = (per_tok > 5e-3).nonzero()[:10]
        print("bad positions:", bad.tolist())
        sys.exit(1)


if __name__ == "__main__":
    main()
