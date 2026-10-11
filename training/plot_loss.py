# -*- coding: utf-8 -*-
"""从 train_log.jsonl 绘制 Loss 曲线 -> loss_curve.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/pretrain"
log = os.path.join(run_dir, "train_log.jsonl")
steps, total, lm = [], [], []
with open(log, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        steps.append(r["step"])
        total.append(r["total"])
        lm.append(r["lm"])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(steps, total, label="total_loss", color="#d62728")
ax.plot(steps, lm, label="lm_loss", color="#1f77b4")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title("LingJie-Code-v1-Pro training loss (%s)" % run_dir)
ax.legend()
ax.grid(alpha=0.3)
out = os.path.join(run_dir, "loss_curve.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
