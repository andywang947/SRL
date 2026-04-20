import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# -------------------------
# Argument
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--input_dir', type=str, default='input')
parser.add_argument('--output_dir', type=str, default='nonrain_mean')
parser.add_argument('--thresh', type=float, default=70.0)
args = parser.parse_args()

# -------------------------
# Path
# -------------------------
input_root = os.path.join('./dataset', args.dataset, args.input_dir)
output_root = os.path.join('./dataset', args.dataset, args.output_dir)
os.makedirs(output_root, exist_ok=True)

files = sorted(os.listdir(input_root))

# -------------------------
# Process
# -------------------------
def compute_non_rain_mask(img, thresh):
    """
    Compute non-rain mask based on weak gradient magnitude.

    Args:
        img (np.ndarray): Input BGR image (H, W, 3).
        thresh (float): Gradient magnitude threshold.

    Returns:
        mask (np.ndarray): Binary mask (H, W), uint8, values in {0, 255}.
    """
    if img is None:
        return None

    # BGR -> Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitude
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # non-rain = weak gradient
    non_rain = mag < thresh
    mask = (non_rain.astype(np.uint8) * 255)

    # Optional: clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def compute_local_mean_mask(img, ksize=5):
    """
    Compute mask by comparing each pixel with its local mean.

    Args:
        img (np.ndarray): Input BGR image (H, W, 3).
        ksize (int): Kernel size for mean filter (default=5).

    Returns:
        mask (np.ndarray): Binary mask (H, W), uint8, values in {0, 255}.
    """
    if img is None:
        return None

    # BGR -> Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 5x5 mean filter
    local_mean = cv2.blur(gray, (ksize, ksize))

    # pixel < local mean
    below_mean = gray < local_mean
    mask = (below_mean.astype(np.uint8) * 255)

    # Optional: clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def extract_nonrain_layer(input_img, blur_ksize=21, sigma=10):
    # 1. convert to gray
    rain = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(rain, (blur_ksize, blur_ksize), sigma)

    # 2. horizontal local mean
    horizontal_kernel = (21, 1)
    local_mean = cv2.blur(rain, horizontal_kernel)
    rain = np.where(rain < local_mean, rain, 0)

    # 3. vertical morphology (keep vertical rain streaks)
    verticalsize = 11
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # rain = cv2.erode(rain, verticalStructure)
    # rain = cv2.dilate(rain, verticalStructure)

    # 4. binarize
    rain = np.where(rain > 0, 255, 0).astype(np.uint8)

    return rain

for name in tqdm(files):
    img = cv2.imread(os.path.join(input_root, name))
    sobel_non_rain_mask = compute_non_rain_mask(img, args.thresh)
    mean_non_rain_mask = compute_local_mean_mask(img)
    non_rain_mask = extract_nonrain_layer(img)

    cv2.imwrite(os.path.join(output_root, name), non_rain_mask)

print(f'[Done] Sobel non-rain masks saved to {output_root}')
