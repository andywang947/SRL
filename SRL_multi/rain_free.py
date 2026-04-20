import cv2
import numpy as np

def rain_free_edge(input_path, output_path):
    # 1. 讀入圖片 (BGR)
    img = cv2.imread(input_path).astype(np.float32) / 255.0

    # 2. 轉成 R,G,B 三通道
    B, G, R = cv2.split(img)
    # 計算每個像素的最小通道
    min_ch = np.minimum(np.minimum(R, G), B)
    print(min_ch)

    # 3. 對每個通道減掉 min(R,G,B)
    R_ = np.clip(R - min_ch, 0, 1)
    G_ = np.clip(G - min_ch, 0, 1)
    B_ = np.clip(B - min_ch, 0, 1)

    # R_ = np.clip(R, 0, 1)
    # G_ = np.clip(G, 0, 1)
    # B_ = np.clip(B, 0, 1)

    # 4. 合併成新影像
    img_sub = cv2.merge([B_, G_, R_])

    # 5. 轉成灰階 (可視為顏色差平均)
    gray = np.mean(img_sub, axis=2)

    # 6. Sobel 邊緣檢測
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)

    # 7. 正規化到 [0,1]
    sobel_mag = sobel_mag / (sobel_mag.max() + 1e-8)

    # 8. 轉為 uint8 並輸出
    sobel_out = (sobel_mag * 255).astype(np.uint8)
    # cv2.imwrite(output_path, sobel_out)
    cv2.imwrite(output_path, (img_sub * 255).astype(np.uint8))


    print(f"✅ Done. Saved to {output_path}")

# 範例使用
if __name__ == "__main__":
    rain_free_edge("1_gt.png", "output_edge.png")
