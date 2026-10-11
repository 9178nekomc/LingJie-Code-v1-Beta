# -*- coding: utf-8 -*-
"""
LingJie-Code 本地对话服务（SSE 流式输出版）
  python web_chat/app.py  ->  http://localhost:8000
零依赖：仅 PyTorch + 标准库。/api/chat 以 Server-Sent Events 推送：
  {"type":"stage", ...}   决策链各关卡事件
  {"type":"token", ...}   逐 token 生成内容
  {"type":"done", ...}    最终结果
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "inference"))

import torch

torch.set_num_threads(max(4, (os.cpu_count() or 8) - 1))
torch.set_grad_enabled(False)

CKPT = os.path.join(ROOT, "artifacts", "ckpt_final.pt")
TOK = os.path.join(ROOT, "tokenizer", "tokenizer.json")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

print("loading LingJie-Code-v1-Pro ...", flush=True)
_t0 = time.time()
from pipeline import LingJiePipeline

PIPE = LingJiePipeline(CKPT, tok_path=TOK, device="cpu", seed=None)
print("ready in %.1fs" % (time.time() - _t0), flush=True)

LOCK = threading.Lock()
HISTORY = []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(
            body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(STATIC, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/history":
            self._send(200, {"history": HISTORY})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(n).decode("utf-8"))["message"]
        except Exception as e:
            return self._send(400, {"error": "bad request: %s" % e})
        msg = (msg or "").strip()
        if not msg:
            return self._send(400, {"error": "empty message"})
        if len(msg) > 500:
            return self._send(400, {"error": "message too long (<=500 chars)"})

        # ---- SSE 流式响应 ----
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def sse(ev):
            try:
                self.wfile.write(("data: %s\n\n" % json.dumps(
                    ev, ensure_ascii=False)).encode("utf-8"))
                self.wfile.flush()
            except Exception:
                raise ConnectionAbortedError

        t0 = time.time()
        try:
            with LOCK:
                r = PIPE.run(msg, emit=sse)
            dt = round(time.time() - t0, 1)
            stages = [{k: t[k] for k in t
                       if k in ("stage", "p_code", "verb", "conf", "meta",
                                "tid", "err", "attempt", "p_consist",
                                "cand_tids") and t.get(k) is not None}
                      for t in r.get("trace", [])
                      if isinstance(t, dict) and "stage" in t]
            resp = {"status": r["status"], "code": r.get("code"),
                    "stdout": r.get("stdout"), "reason": r.get("reason"),
                    "detail": r.get("detail"), "stages": stages,
                    "elapsed": dt}
            HISTORY.append({"q": msg, **resp})
            HISTORY[:] = HISTORY[-50:]
            sse({"type": "done", "result": resp})
        except ConnectionAbortedError:
            pass
        except Exception as e:
            try:
                sse({"type": "error", "error": str(e)})
            except Exception:
                pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("serving -> http://localhost:%d" % port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
