# -*- coding: utf-8 -*-
"""实验：CodeGen-350M-mono 英文 prompt vs 中文 prompt（本地 CPU 版）"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

cg_name = "Salesforce/codegen-350M-mono"
tok = AutoTokenizer.from_pretrained(cg_name, trust_remote_code=True)
model = (AutoModelForCausalLM.from_pretrained(cg_name, trust_remote_code=True)
         .to("cpu").eval())
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# ---- 1) 分词器碎片分析 ----
zh = "帮我算一下列表 [3, 1, 2] 的总和并打印"
zh_ids = tok(zh)["input_ids"]
print("=" * 60)
print("中文 prompt 分词 (%d chars -> %d tokens):" % (len(zh), len(zh_ids)))
print("tokens:", [tok.decode([i]) for i in zh_ids][:25])
en = "compute the sum of the list [3, 1, 2] and print it"
en_ids = tok(en)["input_ids"]
print("英文 prompt 分词 (%d chars -> %d tokens):" % (len(en), len(en_ids)))
print("tokens:", [tok.decode([i]) for i in en_ids][:20])

# ---- 2) 同题英文对照 ----
PAIRS = [
    ("Q1", "打印Hello World", "print Hello World"),
    ("Q2", "用for循环从1到5，每次打印i",
     "use a for loop to print the numbers 1 to 5"),
    ("Q3", "获取用户输入，如果小于18就输出no，否则输出yes",
     "read an integer from input, print 'no' if it is less than 18 else print 'yes'"),
    ("Q4", "帮我算一下列表 [3, 1, 2] 的总和并打印",
     "compute the sum of the list [3, 1, 2] and print it"),
    ("Q5", "把列表 [3, 1, 2] 从小到大排序并打印",
     "sort the list [3, 1, 2] in ascending order and print it"),
    ("Q6", "定义一个加法函数，计算 2 加 3，并打印结果",
     "define a function add(a, b) that returns a + b, then print add(2, 3)"),
    ("Q7", "用while循环打印1到5", "print the numbers 1 to 5 using a while loop"),
    ("Q8", "帮我用Python写一个快速排序算法",
     "implement quicksort in Python"),
]

print("\n" + "=" * 60)
for key, zh_q, en_q in PAIRS:
    print("\n--- %s ---" % key, flush=True)
    for lang, q in [("ZH", zh_q), ("EN", en_q)]:
        ids = tok(q + "\n", return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        out = tok.decode(gen[0, ids["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        print("[%s] %s" % (lang, q))
        print("    ->", repr(out.strip()[:220]), flush=True)
