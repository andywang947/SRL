#!/bin/bash

# 指定 GPU 編號
export CUDA_VISIBLE_DEVICES=0

# 要跑的資料集列表
datasets=("Rain12" "Rain100L" "Rain800" "DDN_SIRR_real" "DDN_SIRR_syn")

# 依序執行
for dataset in "${datasets[@]}"
do
    echo "Running SDRL on dataset: $dataset"
    python sdrl.py --dataset "$dataset" --result_name result --use_aux
done