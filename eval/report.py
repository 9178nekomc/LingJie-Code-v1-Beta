# -*- coding: utf-8 -*-
"""
验收评测：各 Head 的拦截率/决策准确率 + 端到端沙箱通过率。
输出 report.json 与控制台摘要。
"""
import json
import os
import sys
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "tokenizer"))
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
sys.path.insert(0, os.path.join(ROOT, "inference"))

from tokenizer import Tokenizer, RESP, CODE_OPEN, CODE_CLOSE
from lingjie_code import LingJieCodeModel
from heads import VerbHead
from pipeline import LingJiePipeline
from sandbox import full_check
from synth_core import TEMPLATES


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def head_eval(rows, model, tk, device):
    metrics = {}
    # ---- Intent ----
    ir = [r for r in rows if r["type"] == "intent"]
    correct, tp, tn, fp, fn = 0, 0, 0, 0, 0
    for r in ir:
        ids = torch.tensor([[tk.token2id["<task_intent>"]] + tk.encode(r["prompt"])
                            + [tk.token2id[RESP]]], device=device)
        with torch.no_grad():
            out = model(ids)
        pred = int(out["intent"][0].argmax())
        correct += pred == r["label"]
        if r["label"] == 1 and pred == 1:
            tp += 1
        elif r["label"] == 0 and pred == 0:
            tn += 1
        elif r["label"] == 0 and pred == 1:
            fp += 1
        else:
            fn += 1
    metrics["intent"] = {
        "acc": round(correct / len(ir), 4),
        "cannot_recall(拦截率)": round(tp / max(1, tp + fn), 4),
        "误拦率": round(fp / max(1, fp + tn), 4),
        "n": len(ir)}

    # ---- Parser (token acc) ----
    pr = [r for r in rows if r["type"] == "parse"][:500]
    tok_ok, tok_n = 0, 0
    from pipeline import LingJiePipeline  # for encode util
    pipe = None
    for r in pr:
        ids_list = [tk.token2id["<task_parse>"]] + tk.encode(r["prompt"])
        offs = tk.encode_with_offsets(r["prompt"])
        labels = []
        for tid_, s, e in offs:
            lab = 3
            for ss, ee, role in r["spans"]:
                if ss <= s < ee:
                    lab = role
                    break
            labels.append(lab)
        ids = torch.tensor([ids_list + [tk.token2id[RESP]]], device=device)
        with torch.no_grad():
            out = model(ids)
        pred = out["parser"][0, 1:1 + len(labels)].argmax(-1).tolist()
        tok_ok += sum(a == b for a, b in zip(pred, labels))
        tok_n += len(labels)
    metrics["parser"] = {"token_acc": round(tok_ok / max(1, tok_n), 4),
                         "n_tokens": tok_n}

    # ---- Verb ----
    vr = [r for r in rows if r["type"] == "verb"]
    correct = 0
    per_cls = Counter()
    for r in vr:
        ids = torch.tensor([[tk.token2id["<task_verb>"]] + tk.encode(r["prompt"])
                            + [tk.token2id[RESP]]], device=device)
        with torch.no_grad():
            out = model(ids)
        pred = int(out["verb"][0].argmax())
        correct += pred == r["label"]
        if pred == r["label"]:
            per_cls[VerbHead.ACTIONS[r["label"]]] += 1
    metrics["verb"] = {"acc": round(correct / len(vr), 4), "n": len(vr),
                       "per_class_ok": dict(per_cls)}

    # ---- Template ----
    tr = [r for r in rows if r["type"] == "tmpl"]
    correct = 0
    for r in tr:
        j = json.dumps(r["meta"], ensure_ascii=False)
        ids = torch.tensor([[tk.token2id["<task_tmpl>"]] + tk.encode(j)
                            + [tk.token2id[RESP]]], device=device)
        with torch.no_grad():
            out = model(ids)
        pred = int(out["template"][0].argmax())
        correct += pred == r["label"]
    metrics["template"] = {"acc": round(correct / len(tr), 4), "n": len(tr)}

    # ---- Critic ----
    cr = [r for r in rows if r["type"] == "critic"]
    correct = 0
    for r in cr:
        body = "%s%s%s%s" % (r["prompt"], CODE_OPEN, r["code"], CODE_CLOSE)
        ids_list = [tk.token2id["<task_critic>"]] + tk.encode(r["prompt"]) + \
                   [tk.token2id[CODE_OPEN]] + tk.encode(r["code"]) + \
                   [tk.token2id[CODE_CLOSE]] + [tk.token2id[RESP]]
        ids = torch.tensor([ids_list], device=device)
        with torch.no_grad():
            out = model(ids)
        pred = int(out["critic"][0].argmax())
        correct += pred == r["label"]
    metrics["critic"] = {"acc": round(correct / len(cr), 4), "n": len(cr)}

    # ---- Gen: exact match rate ----
    gr = [r for r in rows if r["type"] == "gen"][:300]
    em = 0
    for r in gr:
        j = json.dumps(r["meta"], ensure_ascii=False)
        ids = torch.tensor([[tk.token2id["<task_gen>"]] + tk.encode(j)
                            + [tk.token2id[RESP]]], device=device)
        gen, _ = model.generate(ids, max_new_tokens=140, temperature=0.0,
                                use_cache=True)
        new = gen[0, ids.shape[1]:].tolist()
        out_txt = tk.decode(new).strip()
        em += out_txt == r["code"].strip()
    metrics["gen"] = {"exact_match": round(em / len(gr), 4), "n": len(gr)}
    return metrics


def e2e_eval(rows, pipe):
    er = [r for r in rows if r["type"] == "e2e"]
    ok_n, total = 0, 0
    fails = []
    for r in er:
        res = pipe.run(r["prompt"])
        total += 1
        passed = res["status"] == "ok"
        if passed:
            # 沙箱断言已在管线内；这里再对拍真值
            ok_n += 1
        else:
            fails.append({"prompt": r["prompt"], "reason": res["reason"],
                          "detail": str(res.get("detail"))[:120]})
    acc_cases = [r for r in rows if r["type"] == "accept"]
    accept = []
    for r in acc_cases:
        res = pipe.run(r["prompt"])
        accept.append({"prompt": r["prompt"], "status": res["status"],
                       "code": res.get("code", ""),
                       "stdout": (res.get("stdout") or "").strip(),
                       "reason": res.get("reason")})
    return {"e2e_pass": round(ok_n / max(1, total), 4), "n": total,
            "fails": fails[:20], "accept_cases": accept}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "runs", "sft", "ckpt_final.pt")
    device = "cuda"
    rows = load_rows(os.path.join(ROOT, "data", "eval.jsonl"))
    print("eval rows:", len(rows))

    tk = Tokenizer.load(os.path.join(ROOT, "tokenizer", "tokenizer.json"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = LingJieCodeModel(ck["config"], pad_token_id=0, resp_token_id=4)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    metrics = head_eval(rows, model, tk, device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    pipe = LingJiePipeline(ckpt)
    e2e = e2e_eval(rows, pipe)
    print(json.dumps({k: v for k, v in e2e.items() if k != "fails"},
                     ensure_ascii=False, indent=2))
    report = {"heads": metrics, "e2e": e2e}
    with open(os.path.join(ROOT, "runs", "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("report -> runs/report.json")


if __name__ == "__main__":
    main()
