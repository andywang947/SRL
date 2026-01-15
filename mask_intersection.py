import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--mask_a', type=str, default='ldgp') # ldgp
parser.add_argument('--mask_b', type=str, default='rain_mask_model_pred') # non-rain mask
parser.add_argument('--output', type=str, default='rain_intersection_test')
args = parser.parse_args()

root = os.path.join('./dataset', args.dataset)

dir_a = os.path.join(root, args.mask_a)
dir_b = os.path.join(root, args.mask_b)
out_dir = os.path.join(root, args.output)
os.makedirs(out_dir, exist_ok=True)

files = sorted(os.listdir(dir_a))

for name in tqdm(files):
    path_a = os.path.join(dir_a, name)
    path_b = os.path.join(dir_b, name)

    if not os.path.exists(path_b):
        continue

    mask_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    if mask_a is None or mask_b is None:
        continue

    assert mask_a.shape == mask_b.shape

    # final_mask = np.logical_and(
    #     mask_a > 10,
    #     mask_b == 0
    # ).astype(np.uint8) * 255

    # or version
    final_mask = np.logical_or(
        mask_a > 10,
        mask_b > 10
    ).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(out_dir, name), final_mask)

print(f'[Done] intersection masks saved to {out_dir}')
