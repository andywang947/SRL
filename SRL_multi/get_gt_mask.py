import os
import cv2
import numpy as np
import argparse
from glob import glob

# ============================================================
# 處理一張影像：input - gt，轉成 error mask
# ============================================================
def compute_error_mask(input_img, gt_img, threshold=10):

    # 轉 float
    inp = input_img.astype(np.float32)
    gt  = gt_img.astype(np.float32)

    # 差值（取三通道）
    diff = np.abs(inp - gt)                          # H,W,3
    diff_max = diff.max(axis=2)                     # H,W

    # > threshold = 1，else 0
    mask = (diff_max > threshold).astype(np.uint8)

    # 灰階輸出：0 / 255
    mask_gray = (mask * 255).astype(np.uint8)

    return mask_gray


# ============================================================
# 處理整個資料集
# 資料夾格式：
# ./dataset/{dataset}/input/
# ./dataset/{dataset}/gt/
# ============================================================
def process_dataset(dataset):

    input_dir = f"./dataset/{dataset}/input"
    gt_dir    = f"./dataset/{dataset}/target"
    save_dir  = f"./dataset/{dataset}/error_mask"
    os.makedirs(save_dir, exist_ok=True)

    input_files = sorted(glob(os.path.join(input_dir, "*.*")))

    print(f"Dataset = {dataset}")
    print(f"找到 {len(input_files)} 張圖片\n")

    for idx, img_path in enumerate(input_files):
        name = os.path.basename(img_path)

        gt_path = os.path.join(gt_dir, name)
        if not os.path.exists(gt_path):
            print(f"[Skip] 找不到 GT: {name}")
            continue

        img = cv2.imread(img_path)
        gt  = cv2.imread(gt_path)

        out = compute_error_mask(img, gt)

        save_path = os.path.join(save_dir, name)
        cv2.imwrite(save_path, out)

        print(f"[{idx+1}/{len(input_files)}] Done: {name}")

    print("\n=== 全部處理完成 ===")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="test", help="dataset name")
    args = parser.parse_args()

    process_dataset(args.dataset)
