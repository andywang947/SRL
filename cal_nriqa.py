import pyiqa
import torch
import os
from tqdm import tqdm

# 顯示支援的模型名稱（可選）
# print(pyiqa.list_models())

# 裝置選擇
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 建立 BRISQUE 和 NIQE metric
metric_brisque = pyiqa.create_metric('brisque', device=device)
metric_niqe = pyiqa.create_metric('niqe', device=device)

# 資料夾路徑
img_dir_path = 'dataset/DDN_SIRR_real/result_20251019_mask_loss_channel_consistency_loss_original_loss'
# img_dir_path = 'dataset/DDN_SIRR_real/R2A'
input_files = sorted(os.listdir(img_dir_path))

# 儲存結果
scores_brisque = []
scores_niqe = []

# 主迴圈
for filename in tqdm(input_files):
    input_path = os.path.join(img_dir_path, filename)
    
    try:
        score_brisque = metric_brisque(input_path).item()
        scores_brisque.append(score_brisque)
    except AssertionError:
        print(f"Skipping {input_path} (BRISQUE): image has zero variance")
    
    try:
        score_niqe = metric_niqe(input_path).item()
        scores_niqe.append(score_niqe)
    except AssertionError:
        print(f"Skipping {input_path} (NIQE): image has zero variance")
    print(f"for image {filename}, brisque:{score_brisque}, niqe:{score_niqe}")

# 計算平均
average_brisque = sum(scores_brisque) / len(scores_brisque) if scores_brisque else 0
average_niqe = sum(scores_niqe) / len(scores_niqe) if scores_niqe else 0

# 印出結果
print(f'There are {len(scores_brisque)} images.')
print(f'\nAverage BRISQUE: {average_brisque:.3f}')
print(f'Average NIQE: {average_niqe:.3f}')
