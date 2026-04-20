# 要跑的資料集列表
datasets=("Rain12" "Rain100L" "Rain800" "DDN_SIRR_real" "DDN_SIRR_syn")

# 依序執行
for dataset in "${datasets[@]}"
do
    echo "Running img to masked on dataset: $dataset"
    python img_to_masked.py --dataset "$dataset"
done