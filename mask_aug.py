import cv2
import torch
import numpy as np

def random_drop_connected_components(rain_mask, keep_prob=0.5):
    """
    rain_mask: torch.Tensor, shape [B, 1, H, W], values {0,1} or {0,255}
    return: same shape, torch.Tensor
    """
    device = rain_mask.device
    B, _, H, W = rain_mask.shape

    output_masks = []

    for b in range(B):
        mask = rain_mask[b, 0].detach().cpu().numpy()
        mask = (mask > 0).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(mask)

        new_mask = np.zeros_like(mask)

        for label in range(1, num_labels):  # skip background
            if np.random.rand() < keep_prob:
                new_mask[labels == label] = 1

        output_masks.append(new_mask)

    output_masks = np.stack(output_masks, axis=0)  # [B, H, W]
    output_masks = torch.from_numpy(output_masks).float().unsqueeze(1).to(device)

    return output_masks

import torch
import torch.nn.functional as F

def non_rain_local_mean_7x7(input_img, rain_mask):
    """
    input_img : [B, 3, H, W], float tensor
    rain_mask : [B, 1, H, W], values in {0,1} or [0,1]
    
    return:
        local_mean : [B, 3, H, W]
    """

    B, C, H, W = input_img.shape
    device = input_img.device

    # non-rain mask
    non_rain = 1.0 - rain_mask          # [B,1,H,W]

    # repeat mask to RGB
    non_rain_rgb = non_rain.repeat(1, C, 1, 1)  # [B,3,H,W]

    # masked image
    masked_img = input_img * non_rain_rgb

    # 7x7 box filter
    kernel = torch.ones((C, 1, 7, 7), device=device)

    # sum of non-rain pixels
    sum_img = F.conv2d(
        masked_img,
        kernel,
        padding=3,
        groups=C
    )

    # count of non-rain pixels
    count = F.conv2d(
        non_rain_rgb,
        kernel,
        padding=3,
        groups=C
    )

    # local mean (avoid divide-by-zero)
    local_mean = sum_img / (count + 1e-6)

    return local_mean

import numpy as np
import cv2

def remove_components_if_overlap(mask1, mask2, connectivity=8):
    """
    mask1, mask2: np.ndarray, shape [H, W], values {0,1} or {0,255}
    return: cleaned mask1 (np.uint8, {0,1})
    """
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(m1, connectivity)

    output = m1.copy()

    for label in range(1, num_labels):  # 0 is background
        component = (labels == label)

        # 如果該連通區域和 mask2 有重疊
        if np.any(m2[component]):
            output[component] = 0  # 整塊拔掉

    return output

def remove_components_if_overlap_batch(mask1, mask2):
    """
    mask1, mask2: torch.Tensor [B, 1, H, W], values {0,1}
    return: torch.Tensor [B, 1, H, W]
    """
    device = mask1.device
    B = mask1.shape[0]

    out_masks = []

    for b in range(B):
        m1 = mask1[b, 0].detach().cpu().numpy()
        m2 = mask2[b, 0].detach().cpu().numpy()

        cleaned = remove_components_if_overlap(m1, m2)
        out_masks.append(cleaned)

    out = np.stack(out_masks, axis=0)[:, None, :, :]
    return torch.from_numpy(out).float().to(device)

def random_crop_batch(tensor, crop_h=64, crop_w=64):
    """
    tensor: [B, C, H, W]
    """
    B, C, H, W = tensor.shape
    cropped = []

    for b in range(B):
        top  = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
        cropped.append(
            tensor[b:b+1, :, top:top+crop_h, left:left+crop_w]
        )

    return torch.cat(cropped, dim=0)


def binary_to_distance_strength(rain_mask, eps=1e-6):
    """
    rain_mask: torch.Tensor [B, 1, H, W], binary {0,1}
    return:    torch.Tensor [B, 1, H, W], in [0,1]
    """
    B, _, H, W = rain_mask.shape
    device = rain_mask.device
    out = []

    for b in range(B):
        mask = rain_mask[b, 0].detach().cpu().numpy().astype(np.uint8)
        if mask.sum() == 0:
            out.append(np.zeros_like(mask, dtype=np.float32))
            continue

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist = dist / (dist.max() + eps)
        out.append(dist.astype(np.float32))

    out = torch.from_numpy(np.stack(out)).unsqueeze(1).to(device)
    return out


import numpy as np
import cv2
import random

def shuffle_cc_single(mask_np, allow_overlap=True, max_trials=30):
    """
    mask_np: np.ndarray [H, W], values {0,1} or {0,255}
    return: np.ndarray [H, W]
    """
    H, W = mask_np.shape
    mask_bin = (mask_np > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )

    new_mask = np.zeros_like(mask_bin)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area == 0:
            continue

        comp = (labels[y:y+h, x:x+w] == label).astype(np.uint8)

        for _ in range(max_trials):
            nx = random.randint(0, W - w)
            ny = random.randint(0, H - h)

            if not allow_overlap:
                if np.any(new_mask[ny:ny+h, nx:nx+w] & comp):
                    continue

            new_mask[ny:ny+h, nx:nx+w] |= comp
            break

    return new_mask

def shuffle_connected_components_torch(
    rain_mask,
    allow_overlap=True
):
    """
    rain_mask: torch.Tensor [B, 1, H, W], values {0,1} or {0,255}
    return: torch.Tensor [B, 1, H, W]
    """
    device = rain_mask.device
    B, _, H, W = rain_mask.shape

    rain_mask_np = rain_mask.detach().cpu().numpy()

    out = []
    for b in range(B):
        shuffled = shuffle_cc_single(
            rain_mask_np[b, 0],
            allow_overlap=allow_overlap
        )
        out.append(shuffled)

    out = torch.from_numpy(np.stack(out)).unsqueeze(1)
    return out.to(device).float()

def binary_mask_to_soft(mask, k=7):
    # mask: [B,1,H,W] in {0,1}
    # 1. blur（讓邊界變軟）
    soft = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k//2)

    # 2. normalize 到 [0,1]
    soft = soft / (soft.max(dim=-1, keepdim=True)[0]
                    .max(dim=-2, keepdim=True)[0] + 1e-6)
    return soft