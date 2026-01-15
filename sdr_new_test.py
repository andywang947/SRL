import cv2
import os
import numpy as np
import time
import argparse
from multiprocessing import Pool, cpu_count

# =====================
# Args
# =====================
dataset_name = "test"

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default=f"./dataset/{dataset_name}/input/")
parser.add_argument("--ldgp_path", type=str, default=f"./dataset/{dataset_name}/ldgp/")
parser.add_argument("--save_path", type=str, default=f"./dataset/{dataset_name}/sdr_safe_np/")
parser.add_argument("--fuse_save_path", type=str, default=f"./dataset/{dataset_name}/sdr_safe_fuse_np/")
parser.add_argument("--pow", type=float, default=0.5)
parser.add_argument("--threshold", type=int, default=10)
parser.add_argument("--K", type=int, default=7)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--sdr_num", type=int, default=50)
opt = parser.parse_args()

# =====================
# Paths / params
# =====================
input_path  = opt.input_path
ldgp_path   = opt.ldgp_path
target_path = opt.save_path
target_path2 = opt.fuse_save_path
sdr_num = opt.sdr_num
threshold = opt.threshold

# =====================
# Utils (pure functions only)
# =====================
def make_folder(path):
    os.makedirs(path, exist_ok=True)

def all_rain(mask):
    # 防止 empty slice bug
    if mask.size == 0:
        return False
    return np.all(mask == 1)

# =====================
# Core (NO side effects)
# =====================
def get_mask_and_size(rainy, ldgp_bin, j, i, small_k, center=True):
    """
    IMPORTANT:
    - ldgp_bin must be binary (0/1)
    - NEVER modified inside this function
    """
    H, W = ldgp_bin.shape
    k = small_k

    while True:
        pad = k // 2
        mask = ldgp_bin[
            max(j-pad, 0):min(j+pad+1, H),
            max(i-pad, 0):min(i+pad+1, W)
        ]
        if not all_rain(mask):
            break
        k += 2

    return_mask = np.zeros((k, k), dtype=np.float32)
    return_patch = np.zeros((k, k, 3), dtype=np.float32)

    for y in range(k):
        for x in range(k):
            gy = j + y - pad
            gx = i + x - pad
            if 0 <= gy < H and 0 <= gx < W:
                return_patch[y, x] = rainy[gy, gx]
                if center:
                    return_mask[y, x] = ldgp_bin[gy, gx]

    return return_patch, 1 - return_mask, k

def compute_similarity(rainy, ldgp_bin, j, i):
    neighbors = []
    probs = []

    center_patch, mask, size = get_mask_and_size(
        rainy, ldgp_bin, j, i, opt.k, center=True
    )
    center_vec = (center_patch * mask[..., None]).reshape(-1)

    pad = opt.K // 2
    H, W = ldgp_bin.shape

    for y in range(max(j-pad, 0), min(j+pad+1, H)):
        for x in range(max(i-pad, 0), min(i+pad+1, W)):
            if ldgp_bin[y, x] == 0:
                neigh_patch, _, _ = get_mask_and_size(
                    rainy, ldgp_bin, y, x, size, center=False
                )
                neigh_vec = (neigh_patch * mask[..., None]).reshape(-1)
                dist = np.sum(np.abs(center_vec - neigh_vec))
                prob = 1.0 / max(dist ** opt.pow, 1e-4)
                neighbors.append((y, x))
                probs.append(prob)

    if len(probs) == 0:
        return [], None

    probs = np.array(probs, dtype=np.float32)
    probs /= probs.sum()
    return neighbors, probs

# =====================
# Image-level processing
# =====================
def process_one_image(name):
    np.random.seed(0)   # ← 關鍵這一行
    rainy = cv2.imread(os.path.join(input_path, name))
    ldgp_raw = cv2.imread(os.path.join(ldgp_path, name), cv2.IMREAD_GRAYSCALE)

    # 🔒 唯一一次 threshold
    ldgp_bin = (ldgp_raw > threshold).astype(np.uint8)

    H, W = rainy.shape[:2]
    outputs = np.repeat(rainy[None], sdr_num, axis=0)

    start = time.process_time()
    from tqdm import tqdm
    for j in range(H):
        for i in range(W):
            if ldgp_bin[j, i] == 1:
                neighbors, probs = compute_similarity(rainy, ldgp_bin, j, i)
                if not neighbors:
                    continue
                samples = np.random.choice(len(neighbors), sdr_num, p=probs)
                for n in range(sdr_num):
                    y, x = neighbors[samples[n]]
                    outputs[n, j, i] = rainy[y, x]

    out_dir = os.path.join(target_path, name[:-4])
    make_folder(out_dir)
    for i in range(sdr_num):
        # cv2.imwrite(os.path.join(out_dir, f"{i}.png"), outputs[i])
        cv2.imwrite(os.path.join(out_dir, f"{i}.png"), np.clip(outputs[i], 0, 255).astype(np.uint8))


    fuse = outputs.mean(axis=0)
    fuse_img = np.clip(fuse, 0, 255).astype(np.uint8)
    fuse_path = os.path.join(target_path2,os.path.splitext(name)[0] + ".png")
    cv2.imwrite(fuse_path, fuse_img)

    print(f"[OK] {name} | time: {time.process_time() - start:.2f}s")

# =====================
# Main
# =====================
if __name__ == "__main__":
    make_folder(target_path)
    make_folder(target_path2)

    images = list(set(os.listdir(input_path)) - set(os.listdir(target_path2)))
    workers = min(cpu_count(), 8)

    print(f"Images: {len(images)} | Workers: {workers}")

    with Pool(workers) as p:
        p.map(process_one_image, images)
