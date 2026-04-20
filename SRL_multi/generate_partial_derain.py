import os
import glob
import argparse
import cv2
import numpy as np

def _imread_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def _imread_gray01(path, thr=128):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(f"Cannot read grayscale image: {path}")
    return (g > thr).astype(np.uint8)

def apply_partial_to_input(input_path, partial_dir, sdr_dir, out_dir, thr=128):
    """
    對「單一 name」進行處理：
      - input_path:          e.g. dataset/<DS>/input/andy.png
      - partial_dir:         e.g. dataset/<DS>/partial_ldgp/andy/    (裡面是 1.png,2.png,...)
      - sdr_dir:             e.g. dataset/<DS>/sdr/andy/             (裡面是 1.png,2.png,...)
      - out_dir:             e.g. dataset/<DS>/output/andy/
      - 規則: out = sdr；在 partial==1 的位置，用 input 覆蓋
      - 限制: 尺寸必須一致（input、mask、sdr）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 讀 input（彩色）
    input_img = _imread_color(input_path)
    H, W = input_img.shape[:2]

    # 列出 partial 檔（*.png）
    partial_paths = sorted(glob.glob(os.path.join(partial_dir, "*.png")))
    if not partial_paths:
        raise FileNotFoundError(f"No partial masks found in: {partial_dir}")

    # 對每一個 partial 檔案，找對應的 sdr/<num>.png
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    written = 0

    for p_path in partial_paths:
        num = os.path.splitext(os.path.basename(p_path))[0]  # "1", "2", ...
        sdr_path = os.path.join(sdr_dir, f"{num}.png")
        if not os.path.exists(sdr_path):
            print(f"[WARN] Missing SDR for {base_name}/{num}.png -> {sdr_path}. Skip.")
            continue

        # 讀取 mask 與 sdr
        mask01 = _imread_gray01(p_path, thr=thr)
        sdr_img = _imread_color(sdr_path)

        # 嚴格尺寸檢查（不做 resize）
        if mask01.shape != (H, W):
            raise ValueError(
                f"Size mismatch for mask: {p_path}. Got {mask01.shape}, expected {(H, W)}."
            )
        if sdr_img.shape[:2] != (H, W):
            raise ValueError(
                f"Size mismatch for SDR: {sdr_path}. Got {sdr_img.shape[:2]}, expected {(H, W)}."
            )
        if sdr_img.shape[2] != input_img.shape[2]:
            raise ValueError(
                f"Channel mismatch for SDR vs input: {sdr_path} ({sdr_img.shape}) vs {input_path} ({input_img.shape})."
            )

        # 合成：在 mask==1 的像素，用 input 覆蓋 sdr
        mask = mask01.astype(bool)
        out_img = sdr_img.copy()
        out_img[mask] = input_img[mask]

        out_path = os.path.join(out_dir, f"{num}.png")
        cv2.imwrite(out_path, out_img)
        written += 1

    print(f"Generated {written} outputs for {base_name} -> {out_dir}")

def apply_partial_to_folder(
    root="dataset",
    dataset_name=None,
    input_dir="input",
    partial_root="partial_ldgp",
    sdr_root="sdr",
    out_root="output",
    names=None,
    thr=128
):
    """
    批次處理整個資料夾（每個 name 一套）：
      結構預期：
        {root}/{dataset_name}/input/<name>.png
        {root}/{dataset_name}/partial_ldgp/<name>/{num}.png
        {root}/{dataset_name}/sdr/<name>/{num}.png
        輸出至：
        {root}/{dataset_name}/output/<name>/{num}.png

      - 若 names=None，會自動掃描 partial_ldgp 下的子資料夾作為 name 清單。
      - 尺寸不一致直接拋錯，不自動調整。
    """
    if dataset_name is None:
        raise ValueError("dataset_name is required.")

    ds_root = os.path.join(root, dataset_name)

    dir_input = os.path.join(ds_root, input_dir)
    dir_partial = os.path.join(ds_root, partial_root)
    dir_sdr = os.path.join(ds_root, sdr_root)
    dir_out = os.path.join(ds_root, out_root)

    if not os.path.isdir(dir_input):
        raise FileNotFoundError(f"Input dir not found: {dir_input}")
    if not os.path.isdir(dir_partial):
        raise FileNotFoundError(f"Partial root not found: {dir_partial}")
    if not os.path.isdir(dir_sdr):
        raise FileNotFoundError(f"SDR root not found: {dir_sdr}")
    os.makedirs(dir_out, exist_ok=True)

    # 自動取得 names
    if names is None:
        names = sorted([d for d in os.listdir(dir_partial) if os.path.isdir(os.path.join(dir_partial, d))])
    if not names:
        raise FileNotFoundError(f"No subfolders (names) found in {dir_partial}")

    total_names = 0
    for name in names:
        input_path = os.path.join(dir_input, f"{name}.png")
        partial_dir = os.path.join(dir_partial, name)
        sdr_dir = os.path.join(dir_sdr, name)
        out_dir = os.path.join(dir_out, name)

        if not os.path.exists(input_path):
            input_path = os.path.join(dir_input, f"{name}.jpg")
            if not os.path.exists(input_path):
                print(f"[WARN] Skip {name}: missing input -> {input_path}")
                continue
        if not os.path.isdir(partial_dir):
            print(f"[WARN] Skip {name}: missing partial dir -> {partial_dir}")
            continue
        if not os.path.isdir(sdr_dir):
            print(f"[WARN] Skip {name}: missing sdr dir -> {sdr_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        apply_partial_to_input(
            input_path=input_path,
            partial_dir=partial_dir,
            sdr_dir=sdr_dir,
            out_dir=out_dir,
            thr=thr
        )
        total_names += 1

    print(f"[DONE] Processed {total_names} names under {dir_partial}")

def main():
    parser = argparse.ArgumentParser(description="Apply input pixels onto SDR guided by partial (ldgp) masks.")
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--dataset", default= "test", type=str, help="Dataset name under root.")
    parser.add_argument("--input_dir", type=str, default="input")
    parser.add_argument("--partial_root", type=str, default="partial_ldgp_counterpart")
    parser.add_argument("--sdr_root", type=str, default="sdr")
    parser.add_argument("--out_root", type=str, default="partial_derained_counterpart")
    parser.add_argument("--names", type=str, nargs="*", default=None, help="Optional subset of names to process.")
    parser.add_argument("--thr", type=int, default=128, help="Threshold for binarizing masks (>thr -> 1)")

    args = parser.parse_args()
    apply_partial_to_folder(
        root=args.root,
        dataset_name=args.dataset,
        input_dir=args.input_dir,
        partial_root=args.partial_root,
        sdr_root=args.sdr_root,
        out_root=args.out_root,
        names=args.names,
        thr=args.thr
    )

if __name__ == "__main__":
    main()
