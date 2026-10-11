# -*- coding: utf-8 -*-
"""人工评分 -> R1 风格对比图（图例独立左面板；Cannot 判 0 分）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# Cannot 一律 0 分：只有代码正确才算胜出
SCORES = {
    "打印Hello World\n(基础输出)": {"LingJie": 10, "MiniMind": 2, "CodeGen": 1},
    "for循环打印\n1到5":          {"LingJie": 9,  "MiniMind": 7, "CodeGen": 6},
    "输入+阈值判断\n(复合指令)":    {"LingJie": 10, "MiniMind": 1, "CodeGen": 5},
    "列表求和\n[3,1,2]":          {"LingJie": 10, "MiniMind": 2, "CodeGen": 1},
    "列表升序排序":               {"LingJie": 6,  "MiniMind": 8, "CodeGen": 1},
    "定义加法函数\n并调用":         {"LingJie": 2,  "MiniMind": 8, "CodeGen": 1},
    "while循环\n1到5":            {"LingJie": 5,  "MiniMind": 8, "CodeGen": 3},
    "快速排序\n(超纲,Cannot=0)":   {"LingJie": 0,  "MiniMind": 10, "CodeGen": 0},
}

cats = list(SCORES.keys())
lingjie = [SCORES[c]["LingJie"] * 10 for c in cats]
minimind = [SCORES[c]["MiniMind"] * 10 for c in cats]
codegen = [SCORES[c]["CodeGen"] * 10 for c in cats]
cats_full = cats + ["平均分"]
lingjie.append(round(float(np.mean(lingjie)), 1))
minimind.append(round(float(np.mean(minimind)), 1))
codegen.append(round(float(np.mean(codegen)), 1))

fig = plt.figure(figsize=(16, 6.8))
gs = fig.add_gridspec(1, 2, width_ratios=[0.27, 0.73], wspace=0.06)

# ---- 左侧：独立图例面板（完全隔离） ----
ax_l = fig.add_subplot(gs[0])
ax_l.axis("off")
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="#4C6FE8", hatch="//",
                  edgecolor="white"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#B8B8B8", edgecolor="white"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#C9DCF5", edgecolor="white"),
]
labels = ["LingJie-Code-v1-Pro\n(202M, 从零训练)",
          "MiniMind-3\n(64M, 通用对话)",
          "CodeGen-350M-mono\n(350M, Python专用)"]
ax_l.legend(handles, labels, loc="center left", fontsize=12,
            frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
            handlelength=1.6, handleheight=1.2, labelspacing=1.4,
            borderpad=1.2)
ax_l.set_title("模型图例", fontsize=15, pad=12)

# ---- 右侧：柱状图 ----
ax = fig.add_subplot(gs[1])
x = np.arange(len(cats_full))
w = 0.26
b1 = ax.bar(x - w, lingjie, w, color="#4C6FE8", hatch="//", edgecolor="white")
b2 = ax.bar(x, minimind, w, color="#B8B8B8", edgecolor="white")
b3 = ax.bar(x + w, codegen, w, color="#C9DCF5", edgecolor="white")

for i in range(len(cats_full)):
    vals = [lingjie[i], minimind[i], codegen[i]]
    best = max(vals)
    for j, v in enumerate(vals):
        bar = [b1, b2, b3][j][i]
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, "%g" % v,
                ha="center", va="bottom", fontsize=11,
                fontweight="bold" if v == best else "normal")

ax.set_ylabel("人工评分（满分 100）", fontsize=13)
ax.set_title("Python 基础代码能力人工评测（8 题，Cannot 一律 0 分，代码正确才算胜出；满分 100）",
             fontsize=14, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(cats_full, fontsize=10)
ax.set_ylim(0, 112)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig("bench_chart.png", dpi=140, bbox_inches="tight")
print("saved bench_chart.png")
print("LingJie avg:", lingjie[-1], "MiniMind avg:", minimind[-1],
      "CodeGen avg:", codegen[-1])
