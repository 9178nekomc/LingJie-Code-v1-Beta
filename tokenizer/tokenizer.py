# -*- coding: utf-8 -*-
"""
LingJie-Code-v1-Pro  —  自研 Tokenizer（不依赖任何预训练分词器）

规则：
  1. 数字强制单字切分：1234 -> 1, 2, 3, 4
  2. 缩进空格逐个保留，不合并、不剥离
  3. 词表压缩到 8000~10000：高频整词 + 高频中文单字 + ASCII 字符兜底
  4. 特殊 token：任务前缀 / <resp> / <code> 等
"""
import json
import os
import re
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
RESP = "<resp>"
TASK_TOKENS = ["<task_intent>", "<task_parse>", "<task_verb>",
               "<task_tmpl>", "<task_gen>", "<task_critic>"]
CODE_OPEN, CODE_CLOSE = "<code>", "</code>"

SPECIALS = [PAD, BOS, EOS, UNK, RESP] + TASK_TOKENS + [CODE_OPEN, CODE_CLOSE]

# 单词 = 连续英文字母/下划线；数字永远单切；空格逐个保留
WORD_RE = re.compile(r"[A-Za-z_]+|[0-9]|\s|.", re.DOTALL)


class Tokenizer:
    def __init__(self):
        self.token2id = {}
        self.id2token = []
        self.word2id = {}

    # ------------------------------------------------------------------
    @property
    def vocab_size(self):
        return len(self.id2token)

    def token_id(self, tok):
        return self.token2id.get(tok, self.token2id[UNK])

    # ------------------------------------------------------------------
    def train(self, texts, vocab_cap: int = 9400, min_word_freq: int = 10,
              min_char_freq: int = 5, min_vocab: int = 8192):
        word_freq = Counter()
        char_freq = Counter()
        for t in texts:
            for piece in WORD_RE.findall(t):
                if piece.isspace():
                    continue
                if re.fullmatch(r"[0-9]", piece):
                    continue                       # 数字不进词表（单字常驻）
                if re.fullmatch(r"[A-Za-z_]+", piece):
                    word_freq[piece] += 1
                else:
                    for ch in piece:
                        char_freq[ch] += 1

        self.id2token = list(SPECIALS)
        self.token2id = {t: i for i, t in enumerate(self.id2token)}

        # 高频整词（英文单词/标识符）
        words_by_freq = word_freq.most_common()
        for w, f in words_by_freq:
            if len(self.id2token) >= vocab_cap - 1200:
                break
            if f >= min_word_freq and len(w) >= 2:
                self._add(w)

        # 高频字符（含中文单字），保证任意文本可编码
        for ch, f in char_freq.most_common():
            if len(self.id2token) >= vocab_cap:
                break
            if f >= min_char_freq:
                self._add(ch)

        # ASCII 可打印字符兜底（保证 100% 可编码）
        for code in range(32, 127):
            if len(self.id2token) >= vocab_cap + 200:
                break
            self._add(chr(code))
        self._add("\n")
        # 保留位补齐（封闭域语料达不到 8k 时补占位 token，永不参与编码）
        while len(self.id2token) < min_vocab:
            self._add("<rsv_%d>" % len(self.id2token))
        self.word2id = {w: self.token2id[w] for w in self.id2token
                        if re.fullmatch(r"[A-Za-z_]+", w) and len(w) >= 2}

    def _add(self, tok):
        if tok not in self.token2id:
            self.token2id[tok] = len(self.id2token)
            self.id2token.append(tok)

    # ------------------------------------------------------------------
    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.token2id[BOS])
        for piece in WORD_RE.findall(text):
            if piece.isspace():
                tid = self.token2id.get(piece, self.token2id.get(" ", None))
                if tid is not None:
                    ids.append(tid)          # 每个空格/换行单独成 token
                continue
            if re.fullmatch(r"[0-9]", piece):
                ids.append(self.token2id[piece])     # 数字单字
                continue
            if piece in self.token2id:
                ids.append(self.token2id[piece])
                continue
            # 整词 OOV -> 单字拆分
            for ch in piece:
                ids.append(self.token2id.get(ch, self.token2id[UNK]))
        if add_eos:
            ids.append(self.token2id[EOS])
        return ids

    def encode_with_offsets(self, text):
        """返回 [(token_id, char_start, char_end)]，用于逐 Token 标注对齐"""
        res = []
        ids = self.encode(text)
        i = 0
        for m in WORD_RE.finditer(text):
            piece = m.group()
            start, end = m.span()
            if piece.isspace():
                res.append((ids[i], start, end))
                i += 1
            elif re.fullmatch(r"[0-9]", piece):
                res.append((ids[i], start, end))
                i += 1
            elif piece in self.token2id:
                res.append((ids[i], start, end))
                i += 1
            else:
                for ch in piece:
                    res.append((ids[i], start, start + 1))
                    start += 1
                    i += 1
        assert len(ids) == len(res), \
            f"offset mismatch {len(ids)} vs {len(res)}"
        return res

    def decode(self, ids, skip_special=True):
        toks = []
        specials = set(self.token2id[t] for t in SPECIALS)
        for i in ids:
            if i < 0 or i >= len(self.id2token):
                continue
            if skip_special and i in specials:
                continue
            toks.append(self.id2token[i])
        return "".join(toks)

    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id2token": self.id2token}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        tk = cls()
        with open(path, "r", encoding="utf-8") as f:
            tk.id2token = json.load(f)["id2token"]
        tk.token2id = {t: i for i, t in enumerate(tk.id2token)}
        tk.word2id = {w: tk.token2id[w] for w in tk.id2token
                      if re.fullmatch(r"[A-Za-z_]+", w) and len(w) >= 2}
        return tk


PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
RESP_ID = 4


def task_token_id(tk, name):
    return tk.token2id[f"<task_{name}>"]
