# -*- coding: utf-8 -*-
"""
LingJie-Code-v1-Pro  —  数据集构建
1. 用生成数据训练自研 tokenizer（词表 8000~10000，数字单字，缩进保留）
2. 预分词六层 JSONL -> dataset.npz（ids 拼接 + 元信息 + 各 Head 标签）
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tokenizer"))
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))

from tokenizer import Tokenizer, TASK_TOKENS, RESP, CODE_OPEN, CODE_CLOSE

DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(ROOT, "tokenizer", "tokenizer.json")
OUT = os.path.join(DATA, "dataset.npz")

TASKS = ["intent", "parse", "verb", "tmpl", "gen", "critic"]
TASK_ID = {t: i for i, t in enumerate(TASKS)}
MAX_LEN = 320


def load_jsonl(name):
    path = os.path.join(DATA, "layer_%s.jsonl" % name)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_corpus():
    """tokenizer 训练语料"""
    texts = []
    for name in ["1_intent", "2_parse", "3_verb", "6_critic"]:
        for r in load_jsonl(name):
            texts.append(r["prompt"])
    for name in ["4_template", "5_gen"]:
        for r in load_jsonl(name):
            texts.append(json.dumps(r["meta"], ensure_ascii=False))
            if "code" in r:
                texts.append(r["code"])
    web = os.path.join(DATA, "web_seed.jsonl")
    if os.path.exists(web):
        with open(web, "r", encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line)["text"])
    return texts


def seq_for(tk, task, r):
    """返回 (ids, resp_pos, parser_labels or None, aux dict)"""
    prompt = r.get("prompt", "")
    if task == "intent":
        p = tk.encode(prompt)
        ids = [tk.token2id["<task_intent>"]] + p + \
              [tk.token2id[RESP]] + tk.encode(r["text_label"]) + [2]
        return ids, 1 + len(p), None, {"intent": r["label"]}
    if task == "parse":
        j = json.dumps(r["meta"], ensure_ascii=False)
        p_ids = tk.encode(prompt)
        offs = tk.encode_with_offsets(prompt)
        # 逐 token 标注：token 起点落在哪个 role span
        spans = r["spans"]
        labels = []
        for tid_, s, e in offs:
            lab = 3
            for (ss, ee, role) in spans:
                if ss <= s < ee:
                    lab = role
                    break
            labels.append(lab)
        parser = [-100] + labels
        tail = [tk.token2id[RESP]] + tk.encode(j) + [2]
        assert len(parser) == len(p_ids) + 1
        ids = [tk.token2id["<task_parse>"]] + p_ids + tail
        return ids, len(p_ids) + 1, parser + [-100] * len(tail), {}
    if task == "verb":
        p = tk.encode(prompt)
        ids = [tk.token2id["<task_verb>"]] + p + \
              [tk.token2id[RESP]] + tk.encode(r["text_label"]) + [2]
        return ids, 1 + len(p), None, {"verb": r["label"]}
    if task == "tmpl":
        j = json.dumps(r["meta"], ensure_ascii=False)
        j_ids = tk.encode(j)
        ids = [tk.token2id["<task_tmpl>"]] + j_ids + \
              [tk.token2id[RESP]] + tk.encode(str(r["label"])) + [2]
        return ids, 1 + len(j_ids), None, {"tmpl": r["label"]}
    if task == "gen":
        j = json.dumps(r["meta"], ensure_ascii=False)
        j_ids = tk.encode(j)
        ids = [tk.token2id["<task_gen>"]] + j_ids + \
              [tk.token2id[RESP]] + tk.encode(r["code"]) + [2]
        return ids, 1 + len(j_ids), None, {}
    if task == "critic":
        p = tk.encode(prompt)
        cids = tk.encode(r["code"])
        ids = ([tk.token2id["<task_critic>"]] + p +
               [tk.token2id[CODE_OPEN]] + cids + [tk.token2id[CODE_CLOSE]] +
               [tk.token2id[RESP]] + tk.encode(str(r["label"])) + [2])
        return ids, 1 + len(p) + 1 + len(cids) + 1, None, {"critic": r["label"]}
    raise ValueError(task)


def main():
    # tokenizer 一旦训练过必须复用（词表 ID 与 checkpoint 绑定，重训会导致错位）
    if os.path.exists(TOK_PATH):
        tk = Tokenizer.load(TOK_PATH)
        print("reusing existing tokenizer, vocab:", tk.vocab_size)
    else:
        print("building tokenizer corpus ...")
        corpus = build_corpus()
        print("corpus size:", len(corpus))
        tk = Tokenizer()
        tk.train(corpus, vocab_cap=9400)
        print("vocab size:", tk.vocab_size)
        tk.save(TOK_PATH)

    all_ids, all_parser = [], []
    lens, tasks, resp_pos = [], [], []
    intent_l, verb_l, tmpl_l, critic_l = [], [], [], []

    n_kept, n_drop = 0, 0
    for name in ["1_intent", "2_parse", "3_verb", "4_template", "5_gen",
                 "6_critic"]:
        rows = load_jsonl(name)
        task = {"intent": "intent", "parse": "parse", "verb": "verb",
                "template": "tmpl", "gen": "gen",
                "critic": "critic"}[name.split("_")[1]]
        print("tokenizing", name, len(rows))
        for r in rows:
            ids, rp, parser, aux = seq_for(tk, task, r)
            if len(ids) > MAX_LEN:
                n_drop += 1
                continue
            all_ids.extend(ids)
            lens.append(len(ids))
            tasks.append(TASK_ID[task])
            resp_pos.append(rp)
            intent_l.append(aux.get("intent", -100))
            verb_l.append(aux.get("verb", -100))
            tmpl_l.append(aux.get("tmpl", -100))
            critic_l.append(aux.get("critic", -100))
            if parser is None:
                parser = [-100] * len(ids)
            assert len(parser) == len(ids)
            all_parser.extend(parser)
            n_kept += 1

    print("kept:", n_kept, "dropped(too long):", n_drop)
    np.savez_compressed(
        OUT,
        ids=np.array(all_ids, dtype=np.int32),
        parser=np.array(all_parser, dtype=np.int8),
        lens=np.array(lens, dtype=np.int32),
        task=np.array(tasks, dtype=np.int8),
        resp_pos=np.array(resp_pos, dtype=np.int32),
        intent=np.array(intent_l, dtype=np.int8),
        verb=np.array(verb_l, dtype=np.int8),
        tmpl=np.array(tmpl_l, dtype=np.int8),
        critic=np.array(critic_l, dtype=np.int8),
    )
    print("saved ->", OUT,
          "total tokens:", len(all_ids), "vocab:", tk.vocab_size)


if __name__ == "__main__":
    main()
