# -*- coding: utf-8 -*-
"""随机张量前向传播测试：(2, 64) 输入 -> 模型输出维度验证（L40S）"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

from lingjie_code import LingJieCodeModel


def main():
    device = "cuda"
    model = LingJieCodeModel({"vocab_size": 9600}, pad_token_id=0,
                             resp_token_id=4).to(device)
    n = model.num_parameters()
    print("params: %.2fM (%d)" % (n / 1e6, n))
    assert 150e6 <= n <= 700e6, "param count out of [150M, 700M]"

    x = torch.randint(0, 9600, (2, 64), device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    print("lm_logits:", tuple(out["lm_logits"].shape))
    print("intent:", tuple(out["intent"].shape),
          "parser:", tuple(out["parser"].shape),
          "verb:", tuple(out["verb"].shape),
          "template:", tuple(out["template"].shape),
          "critic:", tuple(out["critic"].shape))
    assert out["lm_logits"].shape == (2, 64, 9600)
    assert out["parser"].shape == (2, 64, 4)
    assert out["intent"].shape == (2, 2)

    # 反向传播 + 梯度检查（LayerScale gamma 是否生效）
    loss = out["lm_logits"].float().mean() + out["intent"].float().mean()
    loss.backward()
    g = model.blocks[0].gamma_attn.grad
    print("gamma_attn grad:", g.abs().mean().item())
    print("gamma init:", model.blocks[0].gamma_attn.detach()[0].item())
    assert abs(model.blocks[0].gamma_attn.detach()[0].item() - 1e-4) < 1e-9
    print("FORWARD TEST PASS")


if __name__ == "__main__":
    main()
