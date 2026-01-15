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
parser.add_argument('--gt_dir', type=str, default='sdr_fuse')
parser.add_argument('--input_dir', type=str, default='input')
parser.add_argument('--output_dir', type=str, default='input_minus_gt_rain')
args = parser.parse_args()

# -------------------------
# Path
# -------------------------
gt_root = os.path.join('./dataset', args.dataset, args.gt_dir)
input_root = os.path.join('./dataset', args.dataset, args.input_dir)
output_root = os.path.join('./dataset', args.dataset, args.output_dir)
os.makedirs(output_root, exist_ok=True)

files = sorted(os.listdir(gt_root))

# -------------------------
# Process
# -------------------------
for fname in tqdm(files):
    gt_path = os.path.join(gt_root, fname)
    input_path = os.path.join(input_root, fname)

    gt = cv2.imread(gt_path)
    inp = cv2.imread(input_path)

    if gt is None or inp is None:
        print(f'[Warning] Missing file: {fname}')
        continue

    # BGR -> Gray
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inp_gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # input - gt，只留下正值（雨）
    rain = inp_gray - gt_gray
    rain = np.clip(rain, 0, 255)

    rain_img = rain.astype(np.uint8)

    out_path = os.path.join(output_root, fname)
    cv2.imwrite(out_path, rain_img)

print(f'[Done] Input-GT rain maps saved to {output_root}')
