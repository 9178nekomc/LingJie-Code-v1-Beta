# -*- coding: utf-8 -*-
"""调试：查看各阶段 LM 生成内容"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "model")
sys.path.insert(0, ".")
from pipeline import LingJiePipeline

pipe = LingJiePipeline(sys.argv[1] if len(sys.argv) > 1 else
                       "runs/pretrain/ckpt_latest.pt", seed=1)
for q in ["打印Hello World", "循环从1到5打印每个数", "如果x大于5就打印big"]:
    print("=" * 50)
    print("Q:", q)
    p, _, _ = pipe.head_probs("intent", q)
    print("intent p_code:", float(p["intent"][0]))
    mt = pipe.gen_after_resp("parse", q, max_new_tokens=90)
    print("parse raw:", repr(mt))
    p, _, _ = pipe.head_probs("verb", q)
    print("verb:", pipe.ACTIONS[int(p["verb"].argmax())],
          round(float(p["verb"].max()), 3))
