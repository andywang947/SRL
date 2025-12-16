import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

from PIL import Image as Image
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from utils import Timer
from network import UNet,Seg_UNet
from data import SDR_dataloader, train_dataloader, Segmentation_dataloader

import torch.nn.functional as F


torch.manual_seed(3)
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="test", help="Dataset name")
parser.add_argument("--result_name", type=str, default="test", help="Dataset name")
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--backbone", type=str, default="Unet", help= "select backbone to be used in SDRL")

opt = parser.parse_args()

opt.rainy_data_path = f"./dataset/{opt.dataset}/"
opt.sdr_data_path = f"./dataset/{opt.dataset}/sdr/"
opt.result_path = f"./dataset/{opt.dataset}/result_{opt.result_name}/"

data_path = opt.rainy_data_path
save_path = opt.result_path
sdr_path = opt.sdr_data_path
epochs = opt.epoch

loss_function = MSELoss()
data_loader = train_dataloader(data_path, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    os.makedirs(save_path)
except:
    pass

epoch_timer = Timer('s') 
total_time = 0

for batch in data_loader:
    try:
        # train 
        rainy_images, sdr_images_target, name, ldgp_img = batch

        h,w = rainy_images.shape[2], rainy_images.shape[3]
        factor = 16

        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        rainy_images = F.pad(rainy_images, (0,padw,0,padh), 'reflect')
        sdr_images_target = F.pad(sdr_images_target, (0,padw,0,padh), 'reflect')
        ldgp_img = F.pad(ldgp_img, (0,padw,0,padh), 'reflect')

        img_save_path = os.path.join(save_path,name[0])
        print(img_save_path)
        if os.path.exists(img_save_path) == True and opt.result_name != "test":
            print("the image exists!")
            continue
        else :
            print("The image now is :", name[0])

        epoch_timer.tic()
        
        model = UNet(is_target=True)
        model = model.to(device)

        # seg_model = UNet(input_channels=3, output_channels=1, is_target=True)
        # seg_model = seg_model.to(device)

        # reconstruct_model = UNet(input_channels=4, output_channels=3, is_target=True)
        # reconstruct_model = reconstruct_model.to(device)

        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # seg_optimizer = Adam(seg_model.parameters(), lr=0.001)
        # seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(seg_optimizer, T_max=epochs)

        # reconstruct_optimizer = Adam(reconstruct_model.parameters(), lr=0.001)
        # reconstruct_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reconstruct_optimizer, T_max=epochs)

        inner_batch_size = 1
        rainy_images = rainy_images.to(device)
        sdr_images_target = sdr_images_target.to(device)
        ldgp_img = ldgp_img.to(device)

        model.train()

        SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=1)

        # for j in tqdm(range(epochs)):
        #     seg_model.train()
        #     for k, inner_batch in enumerate(SDR_loader):
        #         sdr_images, input_img, rain_mask, non_rain_mask, _ = inner_batch
        #         sdr_images = sdr_images.to(device)
        #         input_img = input_img.to(device)
        #         rain_mask = rain_mask.to(device)
        #         non_rain_mask = non_rain_mask.to(device)

        #         with torch.no_grad():
        #             rain_gt = torch.clamp(input_img - sdr_images, min=0.0)
        #             rain_gt = rain_gt.mean(dim=1, keepdim=True)

        #         pred_rain = F.softplus(seg_model(input_img), beta=5)

        #         mask = (rain_mask > 0).float()   # threshold 可調：0.03 ~ 0.1
        #         residual_mask = (rain_gt > 0).float()   # threshold 可調：0.03 ~ 0.1
        #         residual_loss = (torch.abs(pred_rain - rain_gt) * residual_mask).sum() / (residual_mask.sum() + 1e-6)
        #         # residual_loss = (torch.abs(pred_rain - rain_gt) * mask).sum() / (mask.sum() + 1e-6)
        #         rain_loss = (torch.abs(pred_rain - rain_mask) * mask).sum() / (mask.sum() + 1e-6)

        #         # non-rain mask: 1 = non-rain
        #         nr_mask = (non_rain_mask > 0).float()

        #         # 把 pred_rain 當成 activation
        #         # 希望在 non-rain 區域，小於一個 margin
        #         margin = 0.05   # 可調，建議 0.03 ~ 0.1

        #         nonrain_contrastive_loss = (
        #             F.relu(pred_rain - margin) * nr_mask
        #         ).sum() / (nr_mask.sum() + 1e-6)

        #         loss = residual_loss + rain_loss + nonrain_contrastive_loss

        #         seg_optimizer.zero_grad()
        #         loss.backward()
        #         seg_optimizer.step()

        #     seg_scheduler.step()

        #     seg_model.eval()
        #     segnet_output = seg_model(rainy_images)
        #     if opt.result_name == "test":
        #         segnet_output = segnet_output[:, :, :h, :w]

        #         rain = segnet_output[0, 0].detach().cpu().numpy()  # [H, W]
        #         rain = np.clip(rain, 0, 1)
        #         rain = rain / (rain.max() + 1e-6)

        #         plt.imsave(
        #             os.path.join(save_path, "test_seg" + ".png"),
        #             rain,
        #             cmap="gray"
        #         )

        # for j in tqdm(range(epochs)):
        #     reconstruct_model.train()
        #     for k, inner_batch in enumerate(SDR_loader):
        #         sdr_images, input_img, _, _ = inner_batch
        #         sdr_images = sdr_images.to(device)
        #         input_img = input_img.to(device)

        #         segnet_output = seg_model(input_img)

        #         x = torch.cat([sdr_images, segnet_output], dim=1)  # [B, 4, H, W]
        #         pred_input = reconstruct_model(x)
        #         loss = ((pred_input - input_img) ** 2).mean()

        #         reconstruct_optimizer.zero_grad()
        #         loss.backward()
        #         reconstruct_optimizer.step()

        #     reconstruct_scheduler.step()

        #     reconstruct_model.eval()
        #     segnet_output = seg_model(rainy_images)
        #     x = torch.cat([sdr_images_target, segnet_output], dim=1)
        #     reconstructnet_output = reconstruct_model(x)
        #     # if opt.result_name == "test":
        #     #     reconstructnet_output = reconstructnet_output[:,:,:h,:w]
        #     #     reconstructnet_output = np.clip(reconstructnet_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        #     #     plt.imsave(os.path.join(save_path, "test_reconstruct" + str(j) + ".png"), reconstructnet_output)
        import copy
        teacher_model = copy.deepcopy(model)
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.eval()
        # cached_reliable_mask = None


        for j in tqdm(range(epochs)):
            model.train()
            loss_epoch = 0

            for k, inner_batch in enumerate(SDR_loader):
                sdr_images, input_img, rain_mask, non_rain_mask, new_sdr_img = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)
                non_rain_mask = non_rain_mask.to(device)
                new_sdr_img = new_sdr_img.to(device)

                net_output = model(input_img)
                # consistency_output = teacher_model(sdr_images)
                consistency_output = teacher_model(input_img)

                res = torch.abs(net_output - sdr_images)
                consistency = torch.abs(net_output - consistency_output)

                loss = res.mean() + (0.2 * consistency.mean())
                loss_epoch += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # teacher ema
                ema_decay = 0.999
                with torch.no_grad():
                    for param_s, param_t in zip(model.parameters(), teacher_model.parameters()):
                        param_t.data.mul_(ema_decay).add_(param_s.data, alpha=1 - ema_decay)
            scheduler.step()
        
            # inference
            model.eval()
            with torch.no_grad():
                # original
                net_output = model(rainy_images)
                if opt.result_name == "test":
                    net_output = net_output[:,:,:h,:w]
                    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    plt.imsave(os.path.join(save_path,name[0]), denoised)
                    # plt.imsave(os.path.join(save_path,"test_" + str(j) + ".png"), denoised)

        net_output = net_output[:,:,:h,:w]
        denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)

        plt.imsave(os.path.join(save_path,name[0]), denoised)

        time = epoch_timer.toc()
        print("Time: ", time)
        total_time += time
    
    except Exception as e:
        print("Exception occur: ", e)
        pass
    
print("Finish! Average Time:", total_time/len(data_loader))