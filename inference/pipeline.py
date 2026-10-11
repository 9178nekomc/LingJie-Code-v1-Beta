# -*- coding: utf-8 -*-
"""
LingJie-Code-v1-Pro  —  推理管线（Head 级联决策链，硬编码决策逻辑）

用户输入
  -> ① IntentHead   置信度 < 0.75 直接 Cannot（一票否决）
  -> ② 拆解：<task_parse> LM 生成元数据 JSON（ParserHead 逐 Token 标注辅助）
  -> ③ VerbHead 动作路由（CANNOT / 与 JSON.action 不符 -> Cannot/重解析）
  -> ④ TemplateHead 骨架选择（100 模板，校验动作一致性）
  -> ⑤ <task_gen> 元数据 -> 代码
  -> ⑥ 轻量结构校验 + ast 语法检查
  -> ⑦ CriticHead 语义一致性（不一致 -> 修复循环 <= 2）
  -> ⑧ 沙箱断言执行（stdout 对拍模板真值）
  -> 输出代码 / Cannot
"""
import json
import os
import re
import sys

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
from sandbox import full_check, check_syntax
from synth_core import TEMPLATES


class PipelineResult(dict):
    pass


class LingJiePipeline:
    def __init__(self, ckpt_path, tok_path=None, device="cuda",
                 intent_thresh=0.75, max_repairs=2, seed=None):
        self.device = device
        self.torch_device = torch.device(device)
        self.tk = Tokenizer.load(tok_path or os.path.join(
            ROOT, "tokenizer", "tokenizer.json"))
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"])
        self.model = LingJieCodeModel(cfg, pad_token_id=0, resp_token_id=4)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.torch_device).eval()
        self.intent_thresh = intent_thresh
        self.max_repairs = max_repairs
        self.seed = seed
        self.ACTIONS = VerbHead.ACTIONS

    # ------------------------------------------------------------------
    def encode_prompt(self, task, body, until_resp=True):
        """body 中允许嵌入 <code>/<code> 特殊 token，逐段编码"""
        t2i = self.tk.token2id
        ids = [t2i["<task_%s>" % task]]
        parts = re.split(r"(<code>|</code>)", body)
        for seg in parts:
            if seg == CODE_OPEN:
                ids.append(t2i[CODE_OPEN])
            elif seg == CODE_CLOSE:
                ids.append(t2i[CODE_CLOSE])
            elif seg:
                ids.extend(self.tk.encode(seg))
        if until_resp:
            ids.append(t2i[RESP])
        return torch.tensor([ids], device=self.torch_device)

    def head_probs(self, task, body):
        """跑一次前向，返回 (head 概率 dict, 完整输出)"""
        ids = self.encode_prompt(task, body)
        with torch.no_grad():
            out = self.model(ids)
        p = {}
        for k in ["intent", "verb", "template", "critic"]:
            p[k] = F.softmax(out[k].float()[0], dim=-1)
        return p, out, ids

    @torch.no_grad()
    def gen_after_resp(self, task, body, max_new_tokens=120,
                       temperature=0.0, top_k=5):
        """从 <resp> 后自回归生成文本（JAM-DualState Cache 增量）"""
        ids = self.encode_prompt(task, body)
        gen, _ = self.model.generate(
            ids, max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_k=top_k, eos_token_id=2, pad_token_id=0,
            use_cache=True, seed=self.seed)
        new = gen[0, ids.shape[1]:].tolist()
        return self.tk.decode(new)

    # ------------------------------------------------------------------
    def run(self, user_input: str, verbose=False) -> PipelineResult:
        trace = []
        prompt = user_input.strip()

        # ① IntentHead 一票否决
        p, _, _ = self.head_probs("intent", prompt)
        p_code = float(p["intent"][0])
        trace.append({"stage": "intent", "p_code": round(p_code, 4),
                      "threshold": self.intent_thresh})
        if p_code < self.intent_thresh:
            return self._cannot(trace, "intent_reject",
                                "非代码请求或超出能力边界 (p=%.2f)" % p_code)

        # ② 拆解：LM 生成元数据 JSON
        meta_txt = self.gen_after_resp("parse", prompt, max_new_tokens=90)
        meta, meta_err = self._extract_meta(meta_txt)
        trace.append({"stage": "parse", "raw": meta_txt[:200],
                      "meta": meta, "err": meta_err})
        if meta is None:
            return self._cannot(trace, "parse_fail",
                                "无法解析出动作+数据元数据")
        if meta.get("action") == "CANNOT":
            return self._cannot(trace, "parse_cannot", "复杂算法请求，拒绝")

        # ③ VerbHead 动作路由
        p, _, _ = self.head_probs("verb", prompt)
        verb_id = int(p["verb"].argmax())
        verb = self.ACTIONS[verb_id]
        trace.append({"stage": "verb", "verb": verb,
                      "conf": round(float(p["verb"][verb_id]), 4)})
        if verb == "CANNOT":
            return self._cannot(trace, "verb_cannot", "动作路由判定为 CANNOT")
        if verb != meta.get("action"):
            return self._cannot(trace, "verb_mismatch",
                                "动作路由(%s)与拆解动作(%s)不一致"
                                % (verb, meta.get("action")))

        # ④ TemplateHead 骨架选择（Top-5 动作兼容候选，供 ⑤ 代码反选）
        meta_json = json.dumps(meta, ensure_ascii=False)
        p, _, _ = self.head_probs("tmpl", meta_json)
        top = torch.topk(p["template"], 5)
        cand_tids = [int(i) for s, i in zip(top.values, top.indices)
                     if TEMPLATES[int(i)].action == verb]
        trace.append({"stage": "template", "cand_tids": cand_tids,
                      "top5": [(int(i), round(float(s), 3))
                               for s, i in zip(top.values, top.indices)]})
        if not cand_tids:
            return self._cannot(trace, "template_fail", "无匹配代码骨架")

        # ⑤⑦ 生成 + CriticHead 修复循环
        def match_tid(code):
            """用 code_fn(meta) 精确匹配反选模板 ID（消歧同元数据签名模板）"""
            norm = lambda s: "\n".join(l.strip() for l in s.strip().splitlines())
            for t_id in cand_tids:
                try:
                    if norm(TEMPLATES[t_id].code_fn(meta)) == norm(code):
                        return t_id
                except Exception:
                    continue
            return None

        attempt, final_code, final_tid, final_err = 0, None, None, None
        temp = 0.0
        while attempt <= self.max_repairs:
            code_txt = self.gen_after_resp("gen", meta_json,
                                           max_new_tokens=140,
                                           temperature=temp)
            code = self._extract_code(code_txt)
            trace.append({"stage": "gen", "attempt": attempt,
                          "raw": code_txt[:300], "code": code})
            if code is None or not check_syntax(code)[0]:
                final_err = "syntax"
                attempt += 1
                temp = 1.0
                continue
            # CriticHead：语义一致性
            critic_body = "%s%s%s%s" % (prompt, CODE_OPEN, code, CODE_CLOSE)
            ids_c = self.encode_prompt("critic", critic_body)
            with torch.no_grad():
                out_c = self.model(ids_c)
            pc = F.softmax(out_c["critic"].float()[0], dim=-1)
            trace.append({"stage": "critic", "attempt": attempt,
                          "p_consist": round(float(pc[1]), 4)})
            if float(pc[1]) < 0.5:
                final_err = "critic_reject(p=%.2f)" % float(pc[1])
                attempt += 1
                temp = 1.0
                continue
            final_code, final_err = code, None
            final_tid = match_tid(code)
            if final_tid is not None:
                break
            # 代码没对上任何候选骨架：温度重试，最多 max_repairs 次
            final_err = "no_template_match"
            attempt += 1
            temp = 1.0

        if final_code is None:
            return self._cannot(trace, "repair_exhausted",
                                "修复循环 %d 次仍失败: %s"
                                % (self.max_repairs, final_err))

        # ⑥⑧ 结构校验 + 沙箱断言（模板匹配成功 -> stdout 对拍；否则仅执行检查）
        tid = final_tid
        stdin, expected, setup = None, None, None
        if tid is not None:
            try:
                t = TEMPLATES[tid]
                stdin = t.inputs_fn(meta) if t.inputs_fn else None
                expected = t.out_fn(meta) if t.out_fn else None
                setup = t.setup
            except Exception:
                pass
        ok, err, res = full_check(final_code, stdin, expected, setup=setup)
        trace.append({"stage": "sandbox", "ok": ok, "err": (err or "")[:200],
                      "stdout": (res or {}).get("stdout", "")[:100]})
        if not ok:
            # 修复循环：报错触发重新生成，最多 max_repairs 次
            attempt = 0
            temp = 1.0
            while attempt < self.max_repairs:
                code_txt = self.gen_after_resp("gen", meta_json,
                                               max_new_tokens=140,
                                               temperature=temp)
                code = self._extract_code(code_txt)
                trace.append({"stage": "repair", "attempt": attempt,
                              "code": code, "err": (err or "")[:120]})
                if code is None:
                    attempt += 1
                    continue
                m = match_tid(code)
                st = stdin
                ex = expected if m is None else None
                sp = setup
                if m is not None:
                    try:
                        tm = TEMPLATES[m]
                        st = tm.inputs_fn(meta) if tm.inputs_fn else None
                        ex = tm.out_fn(meta) if tm.out_fn else None
                        sp = tm.setup
                    except Exception:
                        pass
                ok2, err2, res2 = full_check(code, st, ex, setup=sp)
                if ok2:
                    final_code, tid = code, m
                    ok, err, res = ok2, err2, res2
                    break
                err = err2
                attempt += 1
            if not ok:
                return self._cannot(trace, "sandbox_fail",
                                    "沙箱断言失败: %s" % (err or ""))
        return PipelineResult({
            "status": "ok", "code": final_code, "stdout": res["stdout"],
            "tid": tid, "verb": verb, "meta": meta, "trace": trace})

    # ------------------------------------------------------------------
    def _extract_meta(self, txt):
        i = txt.find("{")
        j = txt.rfind("}")
        if i < 0 or j <= i:
            return None, "no json"
        try:
            m = json.loads(txt[i:j + 1])
        except Exception as e:
            return None, str(e)
        if not isinstance(m, dict) or "action" not in m:
            return None, "missing action"
        return m, None

    def _extract_code(self, txt):
        # 去掉可能的 markdown 包裹与注释头
        txt = txt.replace("```python", "").replace("```", "")
        lines = [l for l in txt.splitlines()]
        while lines and lines[0].strip() in ("", "python"):
            lines.pop(0)
        code = "\n".join(lines).strip()
        return code if code else None

    def _cannot(self, trace, reason, detail):
        return PipelineResult({"status": "Cannot", "reason": reason,
                               "detail": detail, "trace": trace})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("queries", nargs="*")
    args = ap.parse_args()
    ap_sys_ckpt = args.ckpt
    if not ap_sys_ckpt:
        ap_sys_ckpt = os.path.join(ROOT, "runs", "sft", "ckpt_final.pt")
        if not os.path.exists(ap_sys_ckpt):
            ap_sys_ckpt = os.path.join(ROOT, "runs", "pretrain", "ckpt_final.pt")
    pipe = LingJiePipeline(ap_sys_ckpt, seed=args.seed)
    for q in args.queries or ["打印Hello World",
                              "循环从1到5打印每个数",
                              "今天天气怎么样"]:
        r = pipe.run(q)
        print("=" * 60)
        print("Q:", q)
        print("status:", r["status"])
        if r["status"] == "ok":
            print(r["code"])
            print("stdout:", repr(r["stdout"]))
        else:
            print("reason:", r["reason"], "|", r["detail"])
