import os
import cv2
import numpy as np
import argparse

def apply_sobel_edge_map(input_dir, output_dir):

    if not os.path.exists(input_dir):
        print(f"⚠️ 輸入資料夾不存在：{input_dir}，已跳過，需要有該資料集的 input 圖片才可進行")
        return
    
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        # 處理 .png 檔案
        if filename.lower().endswith(('.png', 'jpg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                print(f"🟡 已存在，跳過：{output_path}")
                continue

            # 讀取為灰階圖
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ 無法讀取圖片：{filename}")
                continue

            # 計算 Sobel 邊緣圖
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # 正規化並轉換型態
            sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
            sobel = sobel.astype(np.uint8)

            # 儲存成 png，保持原檔名
            cv2.imwrite(output_path, sobel)

    print(f"✅ 全部處理完成，邊緣圖已存至：{output_dir}")

# 範例呼叫

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sobel edge maps for a dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name, e.g., Rain12')

    args = parser.parse_args()

    dataset_name = args.dataset
    input_dir = os.path.join('dataset', dataset_name, 'input')
    output_dir = os.path.join('dataset', dataset_name, 'input_edge')

    apply_sobel_edge_map(input_dir, output_dir)