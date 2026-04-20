from PIL import Image
import numpy as np

def apply_transparent_mask(
    image_path: str,
    mask_path: str,
    output_path: str,
    threshold: int = 1
) -> None:
    """
    將 mask 指定的區域設為透明。

    Args:
        image_path: 原圖路徑
        mask_path: rain mask 路徑
        output_path: 輸出 PNG 路徑
        threshold: 二值化門檻，mask >= threshold 視為要透明化
    """

    # 讀取原圖，轉成 RGBA 才能處理透明度
    image = Image.open(image_path).convert("RGBA")

    # 讀取 mask，轉成灰階
    mask = Image.open(mask_path).convert("L")

    # 確保 mask 大小跟原圖一致
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    image_np = np.array(image)
    mask_np = np.array(mask)

    # alpha channel
    alpha = image_np[:, :, 3]

    # mask 白色區域 -> 設為透明
    alpha[mask_np >= threshold] = 0

    # 更新 alpha
    image_np[:, :, 3] = alpha

    # 存檔
    result = Image.fromarray(image_np)
    result.save(output_path)

if __name__ == "__main__":
    apply_transparent_mask(
        image_path="dataset/test_Rain100L_66/input/66.png",
        mask_path="dataset/test_Rain100L_66/ldgp/66.png",
        output_path="output_transparent.png"
    )