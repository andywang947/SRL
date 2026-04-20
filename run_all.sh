#!/bin/bash
trap "echo 'Killing all child processes'; pkill -P $$; exit" SIGINT SIGTERM

# source /home/andy/miniconda3/etc/profile.d/conda.sh
# conda activate SIRI
# 指定 GPU 編號
export CUDA_VISIBLE_DEVICES=0
# 3 4 0 1 5 2

python check_gpu.py

# 要跑的資料集列表
datasets=("Rain12" "Rain100L" "Rain800" "DDN_SIRR_real" "DDN_SIRR_syn" "GT_Rain")

# # 問使用者是否開始訓練
# read -p "Start training? (y/n): " confirm

# if [[ "$confirm" != "y" && "$confirm" != "Y" && "$confirm" != "yes" && "$confirm" != "YES" ]]; then
# echo "Training cancelled."
# exit 0
# fi

# # 依序執行
# for dataset in "${datasets[@]}"
# do
#     echo "Running SDRL on dataset: $dataset"
#     python sdrl.py --dataset "$dataset" --result_name "20251217_color_consistency"
# done
NUM_RUNS=10
for dataset in "${datasets[@]}"
do
    echo "Running SDRL on dataset: $dataset"

    for i in $(seq 1 $NUM_RUNS)
    do
        echo "  -> Run $i"
        python sdrl.py \
            --dataset "$dataset" \
            --result_name "SIRI" &

        # sleep 1
    done

    wait
    echo "Finished dataset: $dataset"
done

# nohup bash run_all.sh > 20260413_standard_L_RVC_with_rain_mask_and_both_crop.log 2>&1 &