# -*- coding: utf-8 -*-
"""
三方对比评测：LingJie-Code-v1-Pro (202M) vs MiniMind-3 (64M) vs CodeGen-350M-mono (25M级~350M)
8 道题，贪心解码，输出全部原文供人工打分。
"""
import json
import os
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "inference")

QUESTIONS = [
    ("Q1_print", "打印Hello World"),
    ("Q2_for", "用for循环从1到5，每次打印i"),
    ("Q3_input_if", "获取用户输入，如果小于18就输出no，否则输出yes"),
    ("Q4_sum", "帮我算一下列表 [3, 1, 2] 的总和并打印"),
    ("Q5_sort", "把列表 [3, 1, 2] 从小到大排序并打印"),
    ("Q6_def", "定义一个加法函数，计算 2 加 3，并打印结果"),
    ("Q7_while", "用while循环打印1到5"),
    ("Q8_oog", "帮我用Python写一个快速排序算法"),
]

device = "cuda"
results = {}

# ---------------- LingJie-Code-v1-Pro ----------------
print(">>> LingJie-Code-v1-Pro ...", flush=True)
from pipeline import LingJiePipeline

pipe = LingJiePipeline("runs/sft3/ckpt_final.pt", seed=1)
lingjie = {}
for key, q in QUESTIONS:
    r = pipe.run(q)
    out = ("[Cannot: %s] %s" % (r["reason"], r.get("detail", ""))
           if r["status"] == "Cannot" else r["code"])
    lingjie[key] = {"prompt": q, "status": r["status"], "output": out}
    print(key, "->", r["status"], flush=True)
results["LingJie-Code-v1-Pro(202M)"] = lingjie
del pipe
torch.cuda.empty_cache()

# ---------------- MiniMind-3 ----------------
print(">>> MiniMind-3 ...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

mm_name = "jingyaogong/minimind-3"
mm_tok = AutoTokenizer.from_pretrained(mm_name, trust_remote_code=True)
mm_model = (AutoModelForCausalLM.from_pretrained(mm_name, trust_remote_code=True)
            .to(device).eval())
mm = {}
for key, q in QUESTIONS:
    msgs = [{"role": "user", "content": q}]
    text = mm_tok.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True)
    ids = mm_tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = mm_model.generate(**ids, max_new_tokens=200, do_sample=False,
                                pad_token_id=mm_tok.eos_token_id)
    out = mm_tok.decode(gen[0, ids["input_ids"].shape[1]:],
                        skip_special_tokens=True)
    mm[key] = {"prompt": q, "output": out.strip()}
    print(key, "done", flush=True)
results["MiniMind-3(64M)"] = mm
del mm_model
torch.cuda.empty_cache()

# ---------------- CodeGen-350M-mono ----------------
print(">>> CodeGen-350M-mono ...", flush=True)
cg_name = "Salesforce/codegen-350M-mono"
cg_tok = AutoTokenizer.from_pretrained(cg_name, trust_remote_code=True)
cg_model = (AutoModelForCausalLM.from_pretrained(cg_name, trust_remote_code=True)
            .to(device).eval())
if cg_tok.pad_token_id is None:
    cg_tok.pad_token = cg_tok.eos_token
cg = {}
for key, q in QUESTIONS:
    prompt = q + "\n"
    ids = cg_tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = cg_model.generate(**ids, max_new_tokens=120, do_sample=False,
                                pad_token_id=cg_tok.eos_token_id)
    out = cg_tok.decode(gen[0, ids["input_ids"].shape[1]:],
                        skip_special_tokens=True)
    cg[key] = {"prompt": q, "output": out.strip()}
    print(key, "done", flush=True)
results["CodeGen-350M-mono(350M,Python专用)"] = cg

with open("bench_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)

with open("bench_outputs.txt", "w", encoding="utf-8") as f:
    for model, rows in results.items():
        f.write("#" * 70 + "\n## " + model + "\n")
        for key, q in QUESTIONS:
            r = rows[key]
            f.write("\n--- %s | %s\n" % (key, q))
            f.write(str(r.get("output", ""))[:800] + "\n")
print("saved bench_outputs.json / bench_outputs.txt", flush=True)
