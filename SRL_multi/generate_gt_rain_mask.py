import os
import cv2
import numpy as np

# === 資料夾設定 ===
base_dir = "dataset/test"
input_dir = os.path.join(base_dir, "input")
gt_dir = os.path.join(base_dir, "target")
output_dir = os.path.join(base_dir, "rainmask_gt")
os.makedirs(output_dir, exist_ok=True)

def is_image(x):
    return x.lower().split(".")[-1] in ["png", "jpg", "jpeg"]

files = sorted([f for f in os.listdir(input_dir) if is_image(f)])
print(f"找到 {len(files)} 張 input 圖片\n")

for filename in files:
    input_path = os.path.join(input_dir, filename)
    gt_path = os.path.join(gt_dir, filename)

    if not os.path.exists(gt_path):
        print(f"[Skip] 缺少 target：{filename}")
        continue

    # 讀圖
    input_img = cv2.imread(input_path)
    gt_img = cv2.imread(gt_path)

    if input_img.shape != gt_img.shape:
        print(f"[Error] 尺寸不同：{filename}")
        continue

    # 轉灰階
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)

    # 相減 (input - target)
    diff = input_gray.astype(np.float32) - gt_gray.astype(np.float32)

    diff_bin = (diff >= 10).astype(np.uint8) * 255  # 存成 0 或 255
    # 正規化到 0~255 方便顯示
    diff_norm = diff - diff.min()
    if diff_norm.max() > 0:
        diff_norm = (diff_norm / diff_norm.max() * 255).astype(np.uint8)
    else:
        diff_norm = np.zeros_like(diff_norm, dtype=np.uint8)

    # 儲存
    out_path = os.path.join(output_dir, filename)
    # cv2.imwrite(out_path, diff_bin)
    cv2.imwrite(out_path, diff_bin)

    print(f"[Save] {out_path}")

print("\n完成！")
