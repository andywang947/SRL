import os
from tqdm import tqdm

# 互動輸入：資料集名稱、目前結果資料夾、baseline 資料夾
dataset = input('請輸入 dataset 名稱（例如：Rain12、Rain100L、Rain800）：').strip()
import pyiqa
import torch

# 設備
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 建立 metrics
metric_psnr = pyiqa.create_metric('psnr', device=device)
metric_ssim = pyiqa.create_metric('ssim', device=device)

current_subdir = "result_20251216_new_loss"
if dataset == "test":
    current_subdir = "result_test"
baseline_subdir = "R2A"

# 路徑
target_dir = f'dataset/{dataset}/target'
current_dir = f'dataset/{dataset}/{current_subdir}'
baseline_dir = f'dataset/{dataset}/{baseline_subdir}'

# 基本檢查
for d in [target_dir, current_dir, baseline_dir]:
    if not os.path.isdir(d):
        raise FileNotFoundError(f'資料夾不存在：{d}')

# 只比「三邊都有」的檔名
current_files = set(os.listdir(current_dir))
baseline_files = set(os.listdir(baseline_dir))
target_files = set(os.listdir(target_dir))
common_files = sorted(list(current_files & baseline_files & target_files))

if not common_files:
    raise RuntimeError('找不到三邊共有的檔案，請確認資料夾與檔名一致。')

# 累計容器
psnr_cur_list, ssim_cur_list = [], []
psnr_base_list, ssim_base_list = [], []

win_psnr_cnt = 0
win_ssim_cnt = 0

# 可選：記錄單張差異（方便想看 top-k）
per_image_stats = []  # (filename, psnr_cur, psnr_base, ssim_cur, ssim_base)

for filename in tqdm(common_files, desc='Evaluating'):
    tgt_path = os.path.join(target_dir, filename)
    cur_path = os.path.join(current_dir, filename)
    base_path = os.path.join(baseline_dir, filename)

    # 計算兩邊對上同一張 target 的分數
    psnr_cur = metric_psnr(cur_path, tgt_path).item()
    ssim_cur = metric_ssim(cur_path, tgt_path).item()

    psnr_base = metric_psnr(base_path, tgt_path).item()
    ssim_base = metric_ssim(base_path, tgt_path).item()

    psnr_cur_list.append(psnr_cur)
    ssim_cur_list.append(ssim_cur)
    psnr_base_list.append(psnr_base)
    ssim_base_list.append(ssim_base)

    difference = psnr_cur - psnr_base
    threshold = 0
    if psnr_cur > psnr_base:
        win_psnr_cnt += 1
        print(f"name: {filename} win, difference = {difference}")
    else:
        print(f"name: {filename} lose, difference = {difference}")
    if ssim_cur > ssim_base:
        win_ssim_cnt += 1

    per_image_stats.append((filename, psnr_cur, psnr_base, ssim_cur, ssim_base))

# 平均＆差值
n = len(common_files)
avg_psnr_cur = sum(psnr_cur_list) / n
avg_ssim_cur = sum(ssim_cur_list) / n
avg_psnr_base = sum(psnr_base_list) / n
avg_ssim_base = sum(ssim_base_list) / n

delta_psnr = avg_psnr_cur - avg_psnr_base
delta_ssim = avg_ssim_cur - avg_ssim_base

# 輸出摘要
print('------------------------------------------------------------')
print(f'共同評測張數：{n} 張（只計算三邊皆存在的影像）')
print(f'目前結果資料夾：{current_dir}')
print(f'Baseline 資料夾：   {baseline_dir}')
print('------------------------------------------------------------')
print(f'[PSNR]   current = {avg_psnr_cur:.3f} | baseline = {avg_psnr_base:.3f} | Δ = {delta_psnr:+.3f}')
print(f'         贏的張數：{win_psnr_cnt}/{n}（{win_psnr_cnt/n*100:.1f}%）')
print(f'[SSIM]   current = {avg_ssim_cur:.4f} | baseline = {avg_ssim_base:.4f} | Δ = {delta_ssim:+.4f}')
print(f'         贏的張數：{win_ssim_cnt}/{n}（{win_ssim_cnt/n*100:.1f}%）')