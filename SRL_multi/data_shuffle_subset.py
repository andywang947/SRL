import os
import random

train_dir = "dataset/RealRain_1k_H_train/input"

# 讀取所有檔名
file_list = sorted(os.listdir(train_dir))  # 先排序，確保一致性
print(file_list)
seed = 42
random.seed(seed)
random.shuffle(file_list)
N = len(file_list)

subset_10 = file_list[:int(0.1 * N)]
subset_20 = file_list[:int(0.2 * N)]
subset_50 = file_list[:int(0.5 * N)]

def save_list(filename, data):
    with open(filename, "w") as f:
        for item in data:
            f.write(item + "\n")

save_list("train_subset_10_filenames.txt", subset_10)
