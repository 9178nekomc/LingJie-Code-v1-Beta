# -*- coding: utf-8 -*-
"""人工评分结果 -> DeepSeek-R1 风格分组柱状图"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 人工逐题打分（0-10），评分人：本次评审（依据 bench_outputs.txt 原始答卷）
# rubric: 意图理解3 + 代码正确可运行3 + 指令细节2 + 规范/得体2
SCORES = {
    "打印Hello World\n(基础输出)": {"LingJie": 10, "MiniMind": 2, "CodeGen": 1},
    "for循环打印\n1到5":          {"LingJie": 9,  "MiniMind": 7, "CodeGen": 6},
    "输入+阈值判断\n(复合指令)":    {"LingJie": 10, "MiniMind": 1, "CodeGen": 5},
    "列表求和\n[3,1,2]":          {"LingJie": 10, "MiniMind": 2, "CodeGen": 1},
    "列表升序排序":               {"LingJie": 6,  "MiniMind": 8, "CodeGen": 1},
    "定义加法函数\n并调用":         {"LingJie": 2,  "MiniMind": 8, "CodeGen": 1},
    "while循环\n1到5":            {"LingJie": 5,  "MiniMind": 8, "CodeGen": 3},
    "快速排序\n(超纲拒绝)":         {"LingJie": 7,  "MiniMind": 10, "CodeGen": 0},
}

cats = list(SCORES.keys())
lingjie = [SCORES[c]["LingJie"] * 10 for c in cats]
minimind = [SCORES[c]["MiniMind"] * 10 for c in cats]
codegen = [SCORES[c]["CodeGen"] * 10 for c in cats]
cats_full = cats + ["平均分"]
lingjie.append(round(np.mean(lingjie), 1))
minimind.append(round(np.mean(minimind), 1))
codegen.append(round(np.mean(codegen), 1))

x = np.arange(len(cats_full))
w = 0.26

fig, ax = plt.subplots(figsize=(15, 6.5))
b1 = ax.bar(x - w, lingjie, w, label="LingJie-Code-v1-Pro (202M, 从零训练)",
            color="#4C6FE8", hatch="//", edgecolor="white")
b2 = ax.bar(x, minimind, w, label="MiniMind-3 (64M)",
            color="#B8B8B8", edgecolor="white")
b3 = ax.bar(x + w, codegen, w, label="CodeGen-350M-mono (350M, Python专用)",
            color="#C9DCF5", edgecolor="white")

for bars in (b1, b2, b3):
    for r in bars:
        pass

# 加粗每组最高分，其余常规标注
for i in range(len(cats_full)):
    vals = [lingjie[i], minimind[i], codegen[i]]
    best = max(vals)
    for j, v in enumerate(vals):
        bar = [b1, b2, b3][j][i]
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, "%g" % v,
                ha="center", va="bottom", fontsize=11,
                fontweight="bold" if v == best else "normal")

ax.set_ylabel("人工评分（满分 100）", fontsize=13)
ax.set_title("Python 基础代码能力人工评测：LingJie-Code-v1-Pro vs MiniMind-3 vs CodeGen-350M-mono\n"
             "（8 题，逐题人工打分：意图理解/正确性/细节/规范，满分 100）",
             fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(cats_full, fontsize=10)
ax.set_ylim(0, 108)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
          fontsize=12, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig("bench_chart.png", dpi=140, bbox_inches="tight")
print("saved bench_chart.png")
