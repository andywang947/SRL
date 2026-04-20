import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# -------------------------
# Argument
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--input_dir', type=str, default='sdr')
parser.add_argument('--output_dir', type=str, default='sdr_fuse')
parser.add_argument('--num_views', type=int, default=50)
args = parser.parse_args()

# -------------------------
# Path
# -------------------------
input_root = os.path.join('./dataset', args.dataset, args.input_dir)
output_root = os.path.join('./dataset', args.dataset, args.output_dir)
os.makedirs(output_root, exist_ok=True)

# 每個子資料夾（1,2,3,...）
folders = sorted(os.listdir(input_root))

# -------------------------
# Process
# -------------------------
for folder in tqdm(folders):
    folder_path = os.path.join(input_root, folder)
    if not os.path.isdir(folder_path):
        continue

    acc = None
    valid_count = 0

    for i in range(0, args.num_views):
        img_path = os.path.join(folder_path, f'{i}.png')
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = img.astype(np.float32)

        if acc is None:
            acc = np.zeros_like(img, dtype=np.float32)

        acc += img
        valid_count += 1

    if valid_count == 0:
        print(f'[Warning] No valid image in {folder}')
        continue

    mean_img = acc / valid_count
    mean_img = np.clip(mean_img, 0, 255).astype(np.uint8)

    out_path = os.path.join(output_root, f'{folder}.png')
    cv2.imwrite(out_path, mean_img)

print(f'[Done] SDR fused images saved to {output_root}')
