# -*- coding: utf-8 -*-
"""
沙箱执行：ast 语法检查 + 受限子进程执行 + stdout 断言。
优先 RestrictedPython（若可用），统一回退到 python -I 隔离子进程（更强隔离）。
"""
import ast
import subprocess
import sys


def check_syntax(code: str):
    """ast 语法检查，返回 (ok, err)"""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, "SyntaxError: %s" % e.msg


DANGEROUS = ("__import__", "eval(", "exec(", "open(/dev", "os.system",
             "subprocess", "shutil", "socket", "while True:\n        pass")


def quick_guard(code: str):
    """轻量静态危险调用拦截（基础代码场景不需要这些）"""
    for d in DANGEROUS:
        if d in code:
            return False, "forbidden construct: %s" % d
    return True, None


def run_code(code: str, stdin_lines=None, timeout=10):
    """在隔离子进程中执行代码。
    返回 dict(ok, stdout, stderr, returncode)"""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            input="\n".join(stdin_lines or []) + ("\n" if stdin_lines else ""),
            capture_output=True, text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout,
                "stderr": proc.stderr, "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "TimeoutExpired",
                "returncode": -9}


def assert_stdout(result, expected: str):
    """stdout 断言（逐行 strip 对比，容忍行尾空白差异）"""
    if expected is None:
        return True, None
    got = "\n".join(l.rstrip() for l in result["stdout"].strip().splitlines())
    exp = "\n".join(l.rstrip() for l in expected.strip().splitlines())
    if got == exp:
        return True, None
    return False, "assert failed: expected %r got %r" % (exp, got)


def full_check(code: str, stdin_lines=None, expected_stdout=None, timeout=10,
               setup=None):
    """ast.parse -> 静态护栏 -> 沙箱执行(可先跑 setup 预置) -> stdout 断言"""
    ok, err = check_syntax(code)
    if not ok:
        return False, err, None
    ok, err = quick_guard(code)
    if not ok:
        return False, err, None
    if setup:
        ok, err = check_syntax(setup)
        if not ok:
            return False, "setup: " + err, None
        code = setup + "\n" + code
    res = run_code(code, stdin_lines, timeout)
    if not res["ok"]:
        return False, res["stderr"] or "exit %d" % res["returncode"], res
    ok, err = assert_stdout(res, expected_stdout)
    if not ok:
        return False, err, res
    return True, None, res
