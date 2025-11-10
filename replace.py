import os
import cv2
import numpy as np

dataset_name = "Rain800"  # 可以改成 "Rain12", "Rain100L", ...

base_dir = "dataset"
input_dir = os.path.join(base_dir, dataset_name, "input")
gt_dir = os.path.join(base_dir, dataset_name, "target")
mask_dir = os.path.join(base_dir, dataset_name, "ldgp")
output_dir = os.path.join(base_dir, dataset_name, "sdr_with_gt")

os.makedirs(output_dir, exist_ok=True)

# 取所有檔名 (假設 input, gt, mask 檔名相同)
file_list = os.listdir(input_dir)

for filename in file_list:
    input_path = os.path.join(input_dir, filename)
    gt_path = os.path.join(gt_dir, filename)
    mask_path = os.path.join(mask_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # 讀取圖片
    input_img = cv2.imread(input_path)  # H, W, C
    gt_img = cv2.imread(gt_path)        # H, W, C
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # H, W

    if input_img is None or gt_img is None or mask is None:
        print(f"跳過 {filename} (檔案缺失或格式錯誤)")
        continue

    # 確保尺寸一致
    if input_img.shape != gt_img.shape or input_img.shape[:2] != mask.shape:
        print(f"尺寸不符，跳過 {filename}")
        continue

    # 把 mask 轉成 0/1
    mask_binary = (mask > 127).astype(np.uint8)  # 白色=1, 黑色=0
    mask_binary_3c = np.repeat(mask_binary[:, :, None], 3, axis=2)

    # 融合
    output = input_img * (1 - mask_binary_3c) + gt_img * mask_binary_3c
    # output = input_img * (1 - mask_binary_3c)
    output = output.astype(np.uint8)

    # 存檔
    cv2.imwrite(output_path, output)
    print(f"已處理 {filename}")
