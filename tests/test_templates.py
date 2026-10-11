# -*- coding: utf-8 -*-
"""模板真值自检：每个模板实例化 N 次，实际执行并断言 stdout 与 out_fn 一致。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
sys.path.insert(0, os.path.join(ROOT, "inference"))

import random
from synth_core import TEMPLATES, make_sample
from sandbox import full_check


def main(n_per=8):
    rng = random.Random(123)
    bad = []
    for tid in sorted(TEMPLATES):
        ok_n = 0
        for _ in range(n_per):
            s = make_sample(tid, rng)
            ok, err, _ = full_check(s["code"], s["inputs"], s["stdout"],
                                    timeout=10, setup=s.get("setup"))
            if ok:
                ok_n += 1
            else:
                bad.append((tid, s["code"], s["stdout"], err))
        print("tid %3d %-8s %d/%d" % (tid, TEMPLATES[tid].action, ok_n, n_per))
    if bad:
        print("\n=== FAILURES ===")
        for tid, code, out, err in bad:
            print("tid", tid, "err:", (err or "")[:150])
            print(code)
            print("expected:", out)
            print("-" * 40)
        sys.exit(1)
    print("ALL TEMPLATES PASS")


if __name__ == "__main__":
    main()
