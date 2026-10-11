# -*- coding: utf-8 -*-
"""复合指令实测"""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "inference")
from pipeline import LingJiePipeline

QUERIES = [
    "获取用户输入，然后判断，如果小于18就输出no，如果大于十八就输出yes",
    "输入一个数，如果大于10打印big，否则打印small",
    "帮我定义一个函数，返回两个数中较大的那个，然后调用打印",
]

pipe = LingJiePipeline(sys.argv[1] if len(sys.argv) > 1 else
                       "runs/sft3/ckpt_final.pt", seed=1)
for q in QUERIES:
    print("=" * 60)
    print("Q:", q)
    r = pipe.run(q)
    print("status:", r["status"])
    if r["status"] == "ok":
        print(r["code"])
        print("stdout:", repr(r["stdout"]))
    else:
        print("reason:", r["reason"], "|", r["detail"])
    for t in r["trace"]:
        if t.get("stage") in ("intent", "parse", "verb", "template", "critic"):
            print(" ", json.dumps(t, ensure_ascii=False)[:200])
