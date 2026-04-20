#!/bin/bash
set -e

DATASETS=("HQ_RAIN")
# DATASETS=("test" "RealRain_1k_H_train" "RealRain_1k_L_train")
# DATASETS=("test" "LHP_rain" "LHP_rain_train")

for DATASET in "${DATASETS[@]}"; do
    echo "================================="
    echo "Running dataset: $DATASET"
    echo "================================="

    python ldgp.py --dataset ${DATASET}
    python sdr_new_test.py --dataset ${DATASET}

done

echo "All datasets finished."

# nohup bash run_pseudo.sh > 20260313_pseudo_HQ_RAIN.log 2>&1 &