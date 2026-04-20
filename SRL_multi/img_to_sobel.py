import cv2
import numpy as np
from matplotlib import pyplot as plt

# 讀取圖片（灰階）
img = cv2.imread('dataset/test/target/1.png', cv2.IMREAD_GRAYSCALE)

# 套用 Sobel Filter
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 合併梯度
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# 正規化為 0~255
sobel_uint8 = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 輸出檔案
cv2.imwrite('sobel_target.png', sobel_uint8)

print("Sobel 邊緣圖已輸出為 sobel_output.png")