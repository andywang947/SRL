PARTIAL_ROOT="partial_ldgp"
OUT_ROOT="partial_derained"
PARTIAL_ROOT_counterpart="partial_ldgp_counterpart"
OUT_ROOT_counterpart="partial_derained_counterpart"

# datasets
DATASETS=(
  "test"
  "Rain12"
  "Rain100L"
  "Rain800"
  "DDN_SIRR_real"
  "DDN_SIRR_syn"
)

# loop to do
for DATASET in "${DATASETS[@]}"; do
  echo "Processing dataset: $DATASET"
  python split_partial_derain.py \
    --dataset "$DATASET"
  python generate_partial_derain.py \
    --dataset "$DATASET" \
    --partial_root "$PARTIAL_ROOT" \
    --out_root "$OUT_ROOT"
  python generate_partial_derain.py \
    --dataset "$DATASET" \
    --partial_root "$PARTIAL_ROOT_counterpart" \
    --out_root "$OUT_ROOT_counterpart"
done

echo "All done!"