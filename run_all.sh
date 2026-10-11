#!/bin/bash
# LingJie-Code-v1-Pro 全链路串行重跑（数据 -> 预训练 -> SFT -> 评测）
set -e
cd /data/lingjie-code
PY=/data/miniconda/envs/torch/bin/python

echo "=== [1/6] regenerate data ==="
$PY data_pipeline/gen_data.py
echo "=== [2/6] build tokenizer + dataset ==="
rm -f tokenizer/tokenizer.json
$PY build_dataset.py
echo "=== [3/6] pretrain ==="
$PY training/pretrain.py --micro-bs 64 --accum 4 --epochs 1.5 --out /data/lingjie-code/runs/pretrain3
echo "=== [4/6] sft ==="
$PY training/sft_train.py --ckpt runs/pretrain3/ckpt_final.pt --micro-bs 64 --accum 4 --out /data/lingjie-code/runs/sft3
echo "=== [5/6] eval ==="
$PY eval/report.py runs/sft3/ckpt_final.pt
echo "=== [6/6] loss curves ==="
$PY training/plot_loss.py runs/pretrain3
$PY training/plot_loss.py runs/sft3
echo "=== ALL DONE ==="
