# -*- coding: utf-8 -*-
"""
从网上拉取开源 Python 基础教程语料（docs.python.org 官方教程等），
提取句式与示例片段，作为合成数据的措辞种子（web_seed.jsonl）。
"""
import json
import os
import re
import sys

import requests

PAGES = [
    "https://docs.python.org/zh-cn/3/tutorial/introduction.html",
    "https://docs.python.org/zh-cn/3/tutorial/controlflow.html",
    "https://docs.python.org/zh-cn/3/tutorial/datastructures.html",
    "https://docs.python.org/zh-cn/3/tutorial/inputoutput.html",
    "https://docs.python.org/zh-cn/3/tutorial/errors.html",
]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "web_seed.jsonl")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for url in PAGES:
        try:
            r = requests.get(url, timeout=20,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            html = r.text
        except Exception as e:
            print("FAIL", url, e)
            continue
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s+", " ", text)
        sents = [s.strip() for s in re.split(r"[。！？!?]", text)
                 if 8 < len(s.strip()) < 120]
        picked = [s for s in sents if re.search(
            r"print|循环|如果|函数|列表|输入|输出|变量|for|while|if", s)][:400]
        for s in picked:
            rows.append({"src": url, "text": s})
        print("OK", url, len(picked), "sentences")
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved", len(rows), "->", OUT)


if __name__ == "__main__":
    main()
