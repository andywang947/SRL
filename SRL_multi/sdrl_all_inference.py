import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from network import UNet
import torch.nn.functional as F_


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== paths =====
input_dir = "./dataset/Rain100L_test/"
save_dir = "./result/inference/"
model_path = "./dataset/Rain100L_train/result_20260403_multi_image_standard/refined_derainer.pth"

os.makedirs(save_dir, exist_ok=True)

# ===== load model =====
model = UNet()
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# ===== get file list =====
image_list = os.listdir(input_dir)
image_list.sort()

# ===== inference =====
with torch.no_grad():
    for name in image_list:
        if not name.endswith(('.png', '.jpg', '.jpeg')):
            continue

        # ===== read image =====
        image = Image.open(os.path.join(input_dir, name)).convert("RGB")
        input_img = F.to_tensor(image).unsqueeze(0).to(device)  # [1, C, H, W]

        h,w = input_img.shape[2], input_img.shape[3]
        factor = 16

        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_img = F_.pad(input_img, (0,padw,0,padh), 'reflect')

        # ===== forward =====
        output = model(input_img)

        # ===== to image =====
        output = output[:,:,:h,:w]
        out = output[0].permute(1, 2, 0).cpu().numpy()
        out = np.clip(out, 0, 1)

        # ===== save =====
        plt.imsave(os.path.join(save_dir, name), out)

print("Inference done")