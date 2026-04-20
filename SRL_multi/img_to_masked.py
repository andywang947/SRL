import os
import cv2
import numpy as np
import argparse

def apply_masked_rgb_save(input_dir, output_dir, mask_dir):

    if not os.path.exists(input_dir):
        print(f"⚠️ 輸入資料夾不存在：{input_dir}，已跳過，需要有該資料集的 input 圖片才可進行")
        return

    if not os.path.exists(mask_dir):
        print(f"⚠️ 遮罩資料夾不存在：{mask_dir}，請提供灰階遮罩圖的資料夾")
        return

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', 'jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            basename = os.path.splitext(filename)[0]
            output_filename = basename + '.png'
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path):
                print(f"🟡 已存在，跳過：{output_path}")
                continue

            # 原圖讀取為 RGB
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"⚠️ 無法讀取圖片：{filename}")
                continue

            # 讀取遮罩圖（灰階）
            if not os.path.exists(mask_path):
                print(f"⚠️ 找不到遮罩圖：{mask_path}，已跳過")
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️ 無法讀取遮罩圖：{mask_path}")
                continue

            if mask.shape != img.shape[:2]:
                print(f"⚠️ 遮罩與輸入圖像尺寸不符：{filename}")
                continue

            # 建立條件遮罩：遮罩值 > 128 的像素位置
            condition = mask > 128

            # 將符合條件的 RGB 值設為 0（黑色）
            img[condition] = [0, 0, 0]

            # 儲存處理後的圖片
            cv2.imwrite(output_path, img)

    print(f"✅ 全部處理完成，邊緣圖已存至：{output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sobel edge maps with masking.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name, e.g., Rain12')

    args = parser.parse_args()

    dataset_name = args.dataset
    input_dir = os.path.join('dataset', dataset_name, 'input')
    mask_dir = os.path.join('dataset', dataset_name, 'ldgp')  # 灰階遮罩圖放這
    output_dir = os.path.join('dataset', dataset_name, 'input_masked')

    apply_masked_rgb_save(input_dir, output_dir, mask_dir)
