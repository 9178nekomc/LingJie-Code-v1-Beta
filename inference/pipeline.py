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
                       temperature=0.0, top_k=5, on_token=None):
        """从 <resp> 后自回归生成文本（JAM-DualState Cache 增量）。
        on_token(累计文本): 每生成一个 token 触发，用于流式输出"""
        ids = self.encode_prompt(task, body)
        acc = []

        def _cb(tok_id):
            acc.append(tok_id)
            if on_token is not None:
                on_token(self.tk.decode(acc))

        gen, _ = self.model.generate(
            ids, max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_k=top_k, eos_token_id=2, pad_token_id=0,
            use_cache=True, seed=self.seed, on_token=_cb)
        return self.tk.decode(acc)

    # ------------------------------------------------------------------
    def run(self, user_input: str, verbose=False, emit=None) -> PipelineResult:
        """emit(event_dict): 可选的流式事件回调（SSE 场景实时推送决策链）"""
        trace = []
        prompt = user_input.strip()

        def emit_stage(name, **kw):
            if emit is not None:
                try:
                    emit({"type": "stage", "stage": name, **kw})
                except Exception:
                    pass

        def emit_tokens(stage):
            """返回一个逐 token 回调，把累计文本作为 token 事件推送"""
            def cb(text):
                if emit is not None:
                    try:
                        emit({"type": "token", "stage": stage, "text": text})
                    except Exception:
                        pass
            return cb

        # ① IntentHead 一票否决
        emit_stage("intent", msg="意图判断中")
        p, _, _ = self.head_probs("intent", prompt)
        p_code = float(p["intent"][0])
        trace.append({"stage": "intent", "p_code": round(p_code, 4),
                      "threshold": self.intent_thresh})
        emit_stage("intent_done", p_code=round(p_code, 4),
                   threshold=self.intent_thresh)
        if p_code < self.intent_thresh:
            emit_stage("cannot", reason="intent_reject",
                       detail="非代码请求或超出能力边界 (p=%.2f)" % p_code)
            return self._cannot(trace, "intent_reject",
                                "非代码请求或超出能力边界 (p=%.2f)" % p_code)

        # ② 拆解：LM 生成元数据 JSON
        emit_stage("parse", msg="拆解动作+数据")
        meta_txt = self.gen_after_resp("parse", prompt, max_new_tokens=90,
                                       on_token=emit_tokens("parse"))
        meta, meta_err = self._extract_meta(meta_txt)
        trace.append({"stage": "parse", "raw": meta_txt[:200],
                      "meta": meta, "err": meta_err})
        if meta is None:
            emit_stage("cannot", reason="parse_fail",
                       detail="无法解析出动作+数据元数据")
            return self._cannot(trace, "parse_fail",
                                "无法解析出动作+数据元数据")
        if meta.get("action") == "CANNOT":
            emit_stage("cannot", reason="parse_cannot",
                       detail="复杂算法请求，拒绝")
            return self._cannot(trace, "parse_cannot", "复杂算法请求，拒绝")

        # ②' 规则层：自然语言区间语义对齐（默认"X到Y/以内"含端点 -> 尾端 +1；
        #     例外：提示词为显式代码 range(a, b) 时原样保留）
        meta_raw = meta
        meta = self._normalize_range(prompt, meta)
        rule_adjusted = meta != meta_raw
        trace.append({"stage": "normalize", "meta_before": meta_raw,
                      "meta_after": meta, "changed": rule_adjusted})

        emit_stage("parse_done", meta=meta, err=meta_err)

        # ③ VerbHead 动作路由
        emit_stage("verb", msg="动作路由中")
        p, _, _ = self.head_probs("verb", prompt)
        verb_id = int(p["verb"].argmax())
        verb = self.ACTIONS[verb_id]
        trace.append({"stage": "verb", "verb": verb,
                      "conf": round(float(p["verb"][verb_id]), 4)})
        emit_stage("verb_done", verb=verb,
                   conf=round(float(p["verb"][verb_id]), 4))
        if verb == "CANNOT":
            emit_stage("cannot", reason="verb_cannot", detail="动作路由判定为 CANNOT")
            return self._cannot(trace, "verb_cannot", "动作路由判定为 CANNOT")
        if verb != meta.get("action"):
            emit_stage("cannot", reason="verb_mismatch",
                       detail="动作路由(%s)与拆解动作(%s)不一致"
                              % (verb, meta.get("action")))
            return self._cannot(trace, "verb_mismatch",
                                "动作路由(%s)与拆解动作(%s)不一致"
                                % (verb, meta.get("action")))

        # ④ TemplateHead 骨架选择（Top-5 动作兼容候选，供 ⑤ 代码反选）
        emit_stage("template", msg="骨架选择中")
        meta_json = json.dumps(meta, ensure_ascii=False)
        p, _, _ = self.head_probs("tmpl", meta_json)
        top = torch.topk(p["template"], 5)
        cand_tids = [int(i) for s, i in zip(top.values, top.indices)
                     if TEMPLATES[int(i)].action == verb]
        trace.append({"stage": "template", "cand_tids": cand_tids,
                      "top5": [(int(i), round(float(s), 3))
                               for s, i in zip(top.values, top.indices)]})
        emit_stage("template_done", cand_tids=cand_tids)
        if not cand_tids:
            emit_stage("cannot", reason="template_fail", detail="无匹配代码骨架")
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
            emit_stage("gen", attempt=attempt, msg="生成代码中",
                       temperature=temp)
            code_txt = self.gen_after_resp(
                "gen", meta_json, max_new_tokens=140, temperature=temp,
                on_token=emit_tokens("gen"))
            code = self._extract_code(code_txt)
            trace.append({"stage": "gen", "attempt": attempt,
                          "raw": code_txt[:300], "code": code})
            emit_stage("gen_done", attempt=attempt, code=code)
            if code is None or not check_syntax(code)[0]:
                final_err = "syntax"
                emit_stage("retry", attempt=attempt, err="syntax",
                           msg="语法错误，修复重试")
                attempt += 1
                temp = 1.0
                continue
            # CriticHead：语义一致性
            emit_stage("critic", attempt=attempt, msg="语义一致性审查中")
            critic_body = "%s%s%s%s" % (prompt, CODE_OPEN, code, CODE_CLOSE)
            ids_c = self.encode_prompt("critic", critic_body)
            with torch.no_grad():
                out_c = self.model(ids_c)
            pc = F.softmax(out_c["critic"].float()[0], dim=-1)
            trace.append({"stage": "critic", "attempt": attempt,
                          "p_consist": round(float(pc[1]), 4)})
            emit_stage("critic_done", attempt=attempt,
                       p_consist=round(float(pc[1]), 4))
            # 规则层已修正区间时跳过 Critic 否决：CriticHead 训练于旧语义数据，
            # 对被规则层纠正后的正确代码会产生系统性误判（伪拒绝信号）
            if float(pc[1]) < 0.5 and not rule_adjusted:
                final_err = "critic_reject(p=%.2f)" % float(pc[1])
                emit_stage("retry", attempt=attempt, err=final_err,
                           msg="语义不一致，修复重试")
                attempt += 1
                temp = 1.0
                continue
            final_code, final_err = code, None
            final_tid = match_tid(code)
            if final_tid is not None:
                break
            # 代码没对上任何候选骨架：温度重试，最多 max_repairs 次
            final_err = "no_template_match"
            emit_stage("retry", attempt=attempt, err=final_err,
                       msg="未匹配任何骨架，修复重试")
            attempt += 1
            temp = 1.0

        if final_code is None:
            emit_stage("cannot", reason="repair_exhausted",
                       detail="修复循环 %d 次仍失败: %s"
                              % (self.max_repairs, final_err))
            return self._cannot(trace, "repair_exhausted",
                                "修复循环 %d 次仍失败: %s"
                                % (self.max_repairs, final_err))

        # ⑥⑧ 结构校验 + 沙箱断言（模板匹配成功 -> stdout 对拍；否则仅执行检查）
        tid = final_tid
        emit_stage("sandbox", msg="沙箱断言执行中", tid=tid)
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
        emit_stage("sandbox_done", ok=ok, err=(err or "")[:200],
                   stdout=(res or {}).get("stdout", ""))
        if not ok:
            # 修复循环：报错触发重新生成，最多 max_repairs 次
            attempt = 0
            temp = 1.0
            while attempt < self.max_repairs:
                emit_stage("repair", attempt=attempt, err=(err or "")[:120],
                           msg="沙箱断言失败，修复重试")
                code_txt = self.gen_after_resp(
                    "gen", meta_json, max_new_tokens=140, temperature=temp,
                    on_token=emit_tokens("gen"))
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
                emit_stage("sandbox_done", ok=ok2, err=(err2 or "")[:200],
                           stdout=(res2 or {}).get("stdout", ""))
                if ok2:
                    final_code, tid = code, m
                    ok, err, res = ok2, err2, res2
                    break
                err = err2
                attempt += 1
            if not ok:
                emit_stage("cannot", reason="sandbox_fail",
                           detail="沙箱断言失败: %s" % (err or ""))
                return self._cannot(trace, "sandbox_fail",
                                    "沙箱断言失败: %s" % (err or ""))
        return PipelineResult({
            "status": "ok", "code": final_code, "stdout": res["stdout"],
            "tid": tid, "verb": verb, "meta": meta, "trace": trace})

    # ------------------------------------------------------------------
    def _normalize_range(self, prompt, meta):
        """
        规则层：自然语言区间语义对齐（默认含端点，尾端 +1）。

        加一（含端点语义）：
          - "从X到Y" / "X到Y" / "X到Y之间" —— 自然语言习惯含两端
          - "Y以内" —— 含边界
        不加一（例外场景，算法判断后保持原样）：
          1. 提示词本身包含显式 Python 代码 range(a, b) —— 用户写的就是字面量，
             严格尊重，绝不修改
          2. 计数语义（"循环N次" / "前N个"）—— 无"到/以内"区间表述，天然不触发
          3. meta.range 的起点与提示词区间起点不一致 —— 防止误伤其他数字
          4. 步长循环含端点同样 +1（0到10步长2 -> range(0, 11, 2) 打到 10）
        """
        if meta.get("action") not in ("for", "sum", "append"):
            return meta
        r = meta.get("range")
        if not r or not isinstance(r, str):
            return meta
        parts = [p.strip() for p in r.split(",")]
        if len(parts) != 2 or not parts[0].lstrip("-").isdigit() \
                or not parts[1].lstrip("-").isdigit():
            return meta
        start, end = int(parts[0]), int(parts[1])

        # 例外 1：提示词是显式代码（用户写的就是字面语义）
        if "range(" in prompt:
            return meta

        # 模式一：自然语言区间 "X到Y"（含端点 -> +1）
        m = re.search(r"(\d+)\s*到\s*(\d+)", prompt)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if start == a and end == b:
                meta = dict(meta)
                meta["range"] = "%d, %d" % (a, b + 1)
            return meta

        # 模式二："Y以内"（含边界 -> +1），要求起点为 0 且终点一致
        m = re.search(r"(\d+)\s*以内", prompt)
        if m and start == 0 and end == int(m.group(1)):
            meta = dict(meta)
            meta["range"] = "0, %d" % (end + 1)
            return meta

        return meta

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
