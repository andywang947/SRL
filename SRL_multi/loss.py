import torch
import torch.nn as nn

class MinRGBPairLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        
    def forward(self, input_img, target_img):
        # input_img, target_img: (B,3,H,W)

        # 1. compute min(R,G,B) for each image
        input_min, _ = torch.min(input_img, dim=1, keepdim=True)     # (B,1,H,W)
        target_min, _ = torch.min(target_img, dim=1, keepdim=True)   # (B,1,H,W)

        # 2. subtract min
        input_minus = input_img - input_min
        target_minus = target_img - target_min

        # 3. L1 between the transformed images
        loss = ((input_minus - target_minus) ** 2)

        return loss

class P50WeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')  # pixel-wise loss

    def forward(self, input_img, output, target):
        # input_img: B x C x H x W
        # output:    B x C x H x W
        # target:    B x C x H x W

        # --- Step 1: pixel-wise loss ---
        loss_map = self.l1(output, target)  # BxCxHxW

        # --- Step 2: compute intensity ---
        # 使用 RGB mean, 你也可以換 min 或 max
        intensity = target.mean(dim=1, keepdim=True)  # Bx1xHxW

        # --- Step 3: 計算每張圖的 Pr50 threshold ---
        # flatten to B x (HW)
        flat = intensity.view(intensity.shape[0], -1)

        # 每張圖算 median (第 50 百分位數)
        # p50 = flat.median(dim=1)[0]  # shape: B
        p50 = flat.mean(dim=1)[0]  # shape: B

        # reshape so broadcasting works
        p50 = p50.view(-1, 1, 1, 1)  # Bx1x1x1

        # --- Step 4: 產生 weight_map: intensity < Pr50 才有效 ---
        weight_map = (intensity < p50).float()  # Bx1xHxW

        # --- Step 5: 套用權重 ---
        weighted_loss = loss_map * weight_map

        # --- Step 6: normalize（避免全 0） ---
        eps = 1e-6
        final_loss = weighted_loss.sum() / (weight_map.sum() * loss_map.shape[1] + eps)

        return final_loss, weight_map

import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalP50WeightedL1Loss(nn.Module):
    def __init__(self, patch_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, input_img, output, target):
        # input_img: B x C x 256 x 256
        B, C, H, W = input_img.shape
        P = self.patch_size  # 64

        # Step 1: pixel-wise loss, shape = B x C x H x W
        loss_map = self.l1(output, target)

        # Step 2: define intensity (可換成 min/max)
        intensity = target.mean(dim=1, keepdim=True)  # B x 1 x H x W

        # Step 3: 切成 patches (每塊 64x64)
        # Unfold: B x C x patch_num x P*P
        int_patches = F.unfold(intensity, kernel_size=P, stride=P)  # B x (1*P*P) x n_patches
        # shape → B x n_patches x (P*P)
        int_patches = int_patches.transpose(1, 2)

        # Step 4: 每個 patch 計算自己的 median
        # median over last dim: B x n_patches
        patch_medians = int_patches.median(dim=2)[0]

        # Step 5: 建立 weight_map 的 patch 版本
        # 對每個 pixel 判斷 < 該 patch 的 median
        # 先將 threshold 展開成每 patch 的 (P*P)
        threshold_expand = patch_medians.unsqueeze(-1)  # B x n_patch x 1
        weight_patches = (int_patches < threshold_expand).float()  # B x n_patch x (P*P)

        # Step 6: fold 回原來的空間 (256×256)
        # transpose 回 B x (patch*patch) x n_patches
        weight_patches = weight_patches.transpose(1, 2)

        # fold: B x 1 x H x W
        weight_map = F.fold(weight_patches, output_size=(H, W),
                            kernel_size=P, stride=P)

        # Step 7: 套用 weighted loss
        weighted_loss = loss_map * weight_map

        eps = 1e-6
        final_loss = weighted_loss.sum() / (weight_map.sum() * C + eps)

        return final_loss, weight_map

def tv_loss(x):
    return torch.mean(torch.abs(x[:, :, :-1] - x[:, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

def channel_consistency_loss(net_output, input_img):
    R_out, G_out, B_out = net_output[:, 0:1, :, :], net_output[:, 1:2, :, :], net_output[:, 2:3, :, :]
    R_ori, G_ori, B_ori = input_img[:, 0:1, :, :], input_img[:, 1:2, :, :], input_img[:, 2:3, :, :]

    # 計算各通道間的差
    out_RG, out_GB, out_BR = R_out - G_out, G_out - B_out, B_out - R_out
    ori_RG, ori_GB, ori_BR = R_ori - G_ori, G_ori - B_ori, B_ori - R_ori

    # 平均三個通道的平方差
    # loss = (
    #     torch.mean((out_RG - ori_RG) ** 2) +
    #     torch.mean((out_GB - ori_GB) ** 2) +
    #     torch.mean((out_BR - ori_BR) ** 2)
    # ) / 3.0

    loss = (
        torch.abs(out_RG - ori_RG) +
        torch.abs(out_GB - ori_GB) +
        torch.abs(out_BR - ori_BR)
    ) / 3.0

    return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalMeanWeightedLoss(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.l1 = nn.L1Loss(reduction='none')
        self.l2 = nn.MSELoss(reduction='none')

    def forward(self, input_img, output, target):
        # input_img: B x C x H x W
        # output:    B x C x H x W
        # target:    B x C x H x W

        # --- Step 1: Pixel-wise loss ---
        loss_map = self.l2(output, target)   # B x C x H x W

        # --- Step 2: Intensity (你可以改 min/max) ---
        intensity = target.mean(dim=1, keepdim=True)  # B x 1 x H x W

        # --- Step 3: 5x5 mean filter ---
        # padding=(kernel_size//2) to keep same spatial size
        # local_mean = F.avg_pool2d(
        #     intensity, 
        #     kernel_size=self.kernel_size, 
        #     stride=1, 
        #     padding=self.kernel_size // 2
        # )   # B x 1 x H x W

        pad = self.kernel_size // 2
        intensity_padded = F.pad(intensity, (pad, pad, pad, pad), mode='reflect')

        local_mean = F.avg_pool2d(
            intensity_padded,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0   # ← padding 已由 F.pad 處理，不需要 avg_pool2d 內建 padding
        )

        # --- Step 4: 產生 weighted map ---
        # difference: 正值代表 intensity < local_mean（你原本的 1）
        diff = local_mean - intensity         # B x 1 x H x W
        # intensity big 

        # softness: 越小越接近硬二值，越大越平滑
        softness = 20.0  

        weight_map = torch.sigmoid(diff * softness)   # B x 1 x H x W
        # weight_map = (intensity < local_mean).float()   # B x 1 x H x W

        # --- Step 5: apply weight ---
        weighted_loss = loss_map * weight_map

        eps = 1e-6
        final_loss = weighted_loss.sum() / (weight_map.sum() * loss_map.shape[1] + eps)

        return final_loss, weight_map

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

def sobel(x):
    # x: [B, C, H, W]
    kernel_x = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        dtype=x.dtype, device=x.device
    ).unsqueeze(0)  # [1,1,3,3]

    kernel_y = torch.tensor(
        [[[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]],
        dtype=x.dtype, device=x.device
    ).unsqueeze(0)

    grad_x = F.conv2d(x, kernel_x.repeat(x.shape[1],1,1,1),
                      padding=1, groups=x.shape[1])
    grad_y = F.conv2d(x, kernel_y.repeat(x.shape[1],1,1,1),
                      padding=1, groups=x.shape[1])

    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

def rain_hf_anti_copy_loss(output, input_img, rain_mask):
    """
    output:    [B,C,H,W]
    input_img:[B,C,H,W]
    rain_mask:[B,1,H,W]  (0/1)
    """

    grad_out = sobel(output)
    grad_in  = sobel(input_img)

    # broadcast mask to channels
    mask = rain_mask.expand_as(grad_out)

    # penalize similarity in high-frequency rain regions
    loss = torch.mean(mask * torch.abs(grad_out - grad_in))
    return loss

def masked_tv_loss(pred, rain_mask):
    # pred: [B,3,H,W], rain_mask: [B,1,H,W]
    dh = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dw = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

    mask_h = rain_mask[:, :, 1:, :]
    mask_w = rain_mask[:, :, :, 1:]

    loss_h = dh * mask_h
    loss_w = dw * mask_w

    return (loss_h.mean() + loss_w.mean())