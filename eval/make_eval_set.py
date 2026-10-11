# -*- coding: utf-8 -*-
"""生成留出评测集（独立种子 777，与训练数据零重叠生成）。"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))

import gen_data as gd
from synth_core import make_sample
import promptgen as pg

OUT = os.path.join(ROOT, "data", "eval.jsonl")

# 重置种子，保证与训练集不同分布实例
gd.rng = random.Random(777)
rng = gd.rng


def main():
    rows = []
    # intent 2000
    for _ in range(1000):
        s, p, spans, meta = gd.base_prompt_sample()
        rows.append({"type": "intent", "prompt": p, "label": 0})
    for _ in range(500):
        rows.append({"type": "intent", "prompt": pg.negative_prompt(rng),
                     "label": 1})
    for _ in range(500):
        rows.append({"type": "intent", "prompt": pg.grey_prompt(rng),
                     "label": 1})
    # parse 1000
    for _ in range(1000):
        s, p, spans, meta = gd.base_prompt_sample()
        rows.append({"type": "parse", "prompt": p, "spans": spans,
                     "meta": meta, "action": s["action"]})
    # verb 1000（覆盖全部 20 类）
    for ai, action in enumerate(gd.ACTIONS[:-1]):
        for _ in range(50):
            tids = [t for t, tt in gd.TEMPLATES.items() if tt.action == action]
            s, p, spans, meta = gd.base_prompt_sample(tid=rng.choice(tids))
            rows.append({"type": "verb", "prompt": p, "label": ai})
    for _ in range(100):
        rows.append({"type": "verb", "prompt": pg.grey_prompt(rng), "label": 19})
    # tmpl 500
    for _ in range(500):
        tid = rng.randrange(100)
        s = make_sample(tid, rng)
        rows.append({"type": "tmpl", "meta": s["meta"], "label": tid})
    # critic 600
    for _ in range(300):
        s, p, spans, meta = gd.base_prompt_sample()
        rows.append({"type": "critic", "prompt": p, "code": s["code"], "label": 1})
    for _ in range(300):
        s, p, spans, meta = gd.base_prompt_sample()
        rows.append({"type": "critic", "prompt": p,
                     "code": gd.mutate_code(s["code"], meta), "label": 0})
    # gen 500（含真值，供代码 BLEU/精确匹配参考）
    for _ in range(500):
        tid = rng.randrange(100)
        s = make_sample(tid, rng)
        rows.append({"type": "gen", "meta": s["meta"], "code": s["code"],
                     "tid": tid})
    # e2e 200（有确定性 stdout 的）
    got = 0
    while got < 200:
        tid = rng.randrange(100)
        s = make_sample(tid, rng)
        if s["stdout"] is None:
            continue
        p, spans, meta = pg.prompt_for_sample(rng, s)
        rows.append({"type": "e2e", "prompt": p, "tid": tid,
                     "meta": s["meta"], "code": s["code"],
                     "stdout": s["stdout"], "inputs": s["inputs"]})
        got += 1
    # 指定验收用例
    rows.append({"type": "accept", "prompt": "打印Hello World",
                 "meta": {"action": "print", "data": ["Hello World"]}})
    rows.append({"type": "accept", "prompt": "用for循环从1到5，每次打印i",
                 "meta": {"action": "for", "range": "1, 5", "data": ["i"],
                          "body": ["print(i)"]}})

    rng.shuffle(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("eval set:", len(rows), "->", OUT)


if __name__ == "__main__":
    main()
