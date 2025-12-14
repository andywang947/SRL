#!/bin/bash

# 指定 GPU 編號
export CUDA_VISIBLE_DEVICES=2

python check_gpu.py

# 要跑的資料集列表
# datasets=("Rain12" "Rain100L" "Rain800" "DDN_SIRR_real" "DDN_SIRR_syn")
datasets=("Rain100L")

# 問使用者是否開始訓練
read -p "Start training? (y/n): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" && "$confirm" != "yes" && "$confirm" != "YES" ]]; then
    echo "Training cancelled."
    exit 0
fi

# 依序執行
for dataset in "${datasets[@]}"
do
    echo "Running SDRL on dataset: $dataset"
    python sdrl.py --dataset "$dataset" --result_name "20251214_test_loss"
done

# nohup bash run_all.sh > 20251214_base_loss.log 2>&1 &