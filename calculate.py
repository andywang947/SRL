import pyiqa
import torch
import os
from tqdm import tqdm

# 檢查是否有 CUDA
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 建立 PSNR metric
iqa_metric_psnr = pyiqa.create_metric('psnr', device=device)
iqa_metric_ssim = pyiqa.create_metric('ssim', device=device)

# 資料夾路徑
dataset = input('請輸入要處理的 dataset 名稱（例如：Rain12、Rain100L、Rain800）：').strip()

input_dir = f'dataset/{dataset}/result_baseline_R2A'
target_dir = f'dataset/{dataset}/target'

# 取得所有檔名（假設兩邊檔名一致）
input_files = sorted(os.listdir(input_dir))

# 儲存所有分數
scores_psnr = []
scores_ssim = []

for filename in tqdm(input_files):
    input_path = os.path.join(input_dir, filename)
    target_path = os.path.join(target_dir, filename)

    # 檢查 target 檔案是否存在
    if not os.path.exists(target_path):
        print(f'Skipping {filename} (no corresponding target)')
        continue

    # 計算 PSNR
    score_psnr = iqa_metric_psnr(input_path, target_path).item()
    score_ssim = iqa_metric_ssim(input_path, target_path).item()
    scores_psnr.append(score_psnr)
    scores_ssim.append(score_ssim)

# 計算平均 PSNR
average_psnr = sum(scores_psnr) / len(scores_psnr) if scores_psnr else 0
average_ssim = sum(scores_ssim) / len(scores_ssim) if scores_ssim else 0
print(f'Average PSNR: {average_psnr:.3f}')
print(f'Average SSIM: {average_ssim:.3f}')