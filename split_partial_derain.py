import os
import argparse
import glob
import numpy as np
import cv2

def load_mask_as_binary(path, thr=128):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return (img > thr).astype(np.uint8)

def save_binary_mask(mask01, path, white_value=255):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = (mask01.astype(np.uint8) * white_value)
    cv2.imwrite(path, out)

def random_partial_by_components(mask01, p=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    num_labels, labels = cv2.connectedComponents((mask01 * 255).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask01), np.zeros_like(mask01)
    keep_flags = rng.random(num_labels) < p
    keep_flags[0] = False  # 背景永遠不保留
    partial = keep_flags[labels].astype(np.uint8) & mask01
    counterpart = (mask01 & (~partial))
    return partial, counterpart

def process_dataset(dataset_name, p=0.5, seed=None, pattern="*.png",
                    n_variants=50, start_idx=1, pad=None):
    input_dir = f"dataset/{dataset_name}/ldgp"
    out_partial_root = f"dataset/{dataset_name}/partial_ldgp"
    out_counter_root = f"dataset/{dataset_name}/partial_ldgp_counterpart"
    os.makedirs(out_partial_root, exist_ok=True)
    os.makedirs(out_counter_root, exist_ok=True)

    exts = [pattern, "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(input_dir, e)))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    # 自動決定序號補零位數（例如 001…）
    if pad is None:
        pad = max(2, len(str(start_idx + n_variants - 1)))

    global_rng = np.random.default_rng(seed)
    total_written = 0

    for path in paths:
        mask01 = load_mask_as_binary(path)
        base = os.path.splitext(os.path.basename(path))[0]

        # 每張圖開自己的子資料夾
        out_partial_dir = os.path.join(out_partial_root, base)
        out_counter_dir = os.path.join(out_counter_root, base)
        os.makedirs(out_partial_dir, exist_ok=True)
        os.makedirs(out_counter_dir, exist_ok=True)

        for k in range(n_variants):
            # 為每個 variant 產生可重現的亂數種子
            per_seed = int(global_rng.integers(0, 2**31 - 1))
            rng = np.random.default_rng(per_seed)

            partial, counterpart = random_partial_by_components(mask01, p=p, rng=rng)
            # print(start_idx)
            # idx = start_idx + k
            fname = f"{k}.png"

            save_binary_mask(partial, os.path.join(out_partial_dir, fname))
            save_binary_mask(counterpart, os.path.join(out_counter_dir, fname))
            total_written += 1

    print(f"Processed {len(paths)} images.")
    print(f"Generated {n_variants} variants per image (total files written: {total_written}).")
    print(f"Partial masks root -> {out_partial_root}")
    print(f"Counterpart masks root -> {out_counter_root}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default= "test", help="資料集名稱，例如 Rain12")
    parser.add_argument("--p", type=float, default=0.5, help="保留每個 component 的機率")
    parser.add_argument("--seed", type=int, default=None, help="隨機種子（控制整體可重現性）")
    parser.add_argument("--n", type=int, default=50, help="每張圖產生的變體張數")
    parser.add_argument("--start_idx", type=int, default=1, help="輸出檔名起始編號")
    parser.add_argument("--pad", type=int, default=None, help="序號左側補零位數（預設自動計算）")
    args = parser.parse_args()

    process_dataset(
        args.dataset, p=args.p, seed=args.seed,
        n_variants=args.n, start_idx=args.start_idx, pad=args.pad
    )

if __name__ == "__main__":
    main()
