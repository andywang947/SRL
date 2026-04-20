import os
import shutil

# ====== 路徑設定 ======
txt_path = "train_subset_10_filenames.txt"          # 你的 txt
source_dir = "dataset/RealRain_1k_H_train/target"               # 原始圖片資料夾
target_dir = "dataset/RealRain_1k_H_train_10_percent/target"      # 目標資料夾

# 建立目標資料夾（如果不存在）
os.makedirs(target_dir, exist_ok=True)

# ====== 讀取檔名 ======
with open(txt_path, "r") as f:
    filenames = [line.strip() for line in f.readlines()]

# ====== 複製檔案 ======
for name in filenames:
    src_path = os.path.join(source_dir, name)
    dst_path = os.path.join(target_dir, name)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {name}")
    else:
        print(f"Not found: {name}")

print("Done.")
