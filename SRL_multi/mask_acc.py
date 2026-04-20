import os
import cv2
import numpy as np
from glob import glob
import argparse

# ============================================================
# 單張影像的評估：precision/recall/F1/IoU/accuracy
# ============================================================
def evaluate_mask(pred_path, target_path, threshold=10):
    pred = cv2.imread(pred_path, 0)
    target = cv2.imread(target_path, 0)

    # binarize to 0/1
    pred_bin   = (pred > threshold).astype(np.uint8)
    target_bin = (target > threshold).astype(np.uint8)

    # TP, FP, FN, TN
    TP = np.sum((pred_bin == 1) & (target_bin == 1))
    FP = np.sum((pred_bin == 1) & (target_bin == 0))
    FN = np.sum((pred_bin == 0) & (target_bin == 1))
    TN = np.sum((pred_bin == 0) & (target_bin == 0))

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy
    }


# ============================================================
# 對整個資料集評估：輸入 pred/ target 資料夾
# ============================================================
def evaluate_dataset(dataset, threshold=10):

    # pred_dir   = f"./dataset/{dataset}/nonrain_sobel"
    # pred_dir   = f"./dataset/{dataset}/nonrain_intersection"
    # pred_dir   = f"./dataset/{dataset}/rain_intersection_test"
    # pred_dir   = f"./dataset/{dataset}/ldgp"
    pred_dir   = f"./dataset/{dataset}/rain_mask_model_pred"
    # pred_dir   = f"./dataset/{dataset}/ldgp_20251216"
    target_dir = f"./dataset/{dataset}/rainmask_gt"
    # target_dir = f"./dataset/{dataset}/non_rain_mask"
    print(f"pred_dir = {pred_dir}")

    pred_files = sorted(glob(os.path.join(pred_dir, "*.*")))

    if len(pred_files) == 0:
        print(f"[Error] 找不到預測影像: {pred_dir}")
        return

    print(f"Dataset = {dataset}")
    print(f"找到 {len(pred_files)} 張預測影像\n")

    # 全資料集的累計分數
    total = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "iou": 0.0,
        "accuracy": 0.0
    }

    count = 0

    for idx, pred_path in enumerate(pred_files):
        name = os.path.basename(pred_path)
        target_path = os.path.join(target_dir, name)

        if not os.path.exists(target_path):
            print(f"[Skip] 找不到 GT: {name}")
            continue

        scores = evaluate_mask(pred_path, target_path, threshold)

        # 累加
        for k in total.keys():
            total[k] += scores[k]

        count += 1

        # print(f"[{idx+1}/{len(pred_files)}] Evaluated: {name}")
        # print(f"   precision={scores['precision']:.4f}, recall={scores['recall']:.4f}, f1={scores['f1']:.4f}")

    # 平均
    avg = {k: (total[k] / count) for k in total.keys()}

    print("\n=== 全資料集平均分數 ===")
    print(f"precision = {avg['precision']:.6f}")
    print(f"recall    = {avg['recall']:.6f}")
    print(f"f1        = {avg['f1']:.6f}")
    print(f"iou       = {avg['iou']:.6f}")
    print(f"accuracy  = {avg['accuracy']:.6f}")

    return avg


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="test", help="dataset name")
    parser.add_argument("--threshold", type=int, default=10, help="binary threshold")
    args = parser.parse_args()

    evaluate_dataset(args.dataset, args.threshold)
