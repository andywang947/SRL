#!/bin/bash
trap "echo 'Killing all child processes'; pkill -P $$; exit" SIGINT SIGTERM

source /home/andy/miniconda3/etc/profile.d/conda.sh
conda activate torch_py312
# 指定 GPU 編號
export CUDA_VISIBLE_DEVICES=5
# 3 4 0 1 5 2

python check_gpu.py

# 要跑的資料集列表
dataset="Rain100L_train"

python sdrl_all.py \
    --dataset "$dataset" \
    --result_name "20260403_multi_image_standard" &

# nohup bash run_all.sh > 20260403_multi_image_standard.log 2>&1 &