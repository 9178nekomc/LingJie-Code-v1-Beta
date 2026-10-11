# -*- coding: utf-8 -*-
"""汇总最终评测 + 六层样本展示"""
import json

r = json.load(open("runs/report.json"))
print(json.dumps(r["heads"], ensure_ascii=False, indent=1))
print("e2e_pass:", r["e2e"]["e2e_pass"], "n =", r["e2e"]["n"])
print("top fails:", json.dumps(r["e2e"]["fails"][:5], ensure_ascii=False)[:500])
print("---- 六层样本展示 ----")
for name in ["1_intent", "2_parse", "3_verb", "4_template", "5_gen", "6_critic"]:
    with open("data/layer_%s.jsonl" % name, encoding="utf-8") as f:
        row = json.loads(f.readline())
    print("[%s] %s" % (name, json.dumps(row, ensure_ascii=False)[:260]))
