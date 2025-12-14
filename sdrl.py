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
        if os.path.exists(img_save_path) == True :
            print("the image exists!")
            continue
        else :
            print("The image now is :", name[0])

        epoch_timer.tic()
        
        model = UNet(is_target=True)
        model = model.to(device)

        def load_encoder_only(model, ckpt_path):
            print("warning: only loading encoder weights")
            ckpt = torch.load(ckpt_path)

            new_ckpt = {}
            for k, v in ckpt.items():
                if "enc_" in k:
                    new_ckpt[k] = v
            # strict=False → 不要求 checkpoint 必須包含所有權重
            model.load_state_dict(new_ckpt, strict=False)

        # load_encoder_only(model, "train_on_Rain100L_epoch1000.pth")
        # model.freeze_encoder()
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


        inner_batch_size = 1
        rainy_images = rainy_images.to(device)
        sdr_images_target = sdr_images_target.to(device)
        ldgp_img = ldgp_img.to(device)

        model.train()

        SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=1)

        best_epoch = 0
        best_val_loss = float('inf')

        for j in tqdm(range(epochs)):
            model.train()
            loss_epoch = 0

            for k, inner_batch in enumerate(SDR_loader):
                sdr_images, input_img, rain_mask, non_rain_mask = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)
                non_rain_mask = non_rain_mask.to(device)

                net_output = model(input_img)
                pixel_loss = ((net_output - sdr_images) ** 2)
                masked_loss = (pixel_loss * rain_mask).sum() / ((rain_mask).sum() + 1e-6)
                # masked_loss = pixel_loss * (1 - non_rain_mask)
                # masked_loss = masked_loss.sum() / ((1 - non_rain_mask).sum() + 1e-6)


                self_loss = ((net_output - input_img) ** 2)
                non_rain_loss = self_loss * non_rain_mask
                non_rain_loss = non_rain_loss.sum() / ((non_rain_mask).sum() + 1e-6)

                uncertain_mask = (rain_mask == 0) & (non_rain_mask == 0)
                uncertain_mask = uncertain_mask.float()
                uncertain_loss = self_loss * uncertain_mask
                uncertain_loss = uncertain_loss.sum() / ((uncertain_mask).sum() + 1e-6)

                # loss = masked_loss + non_rain_loss

                # min_rgb_loss = (feat_loss(net_output, input_img) * (1 - rain_mask)).mean()

                # loss_map = pixel_loss * (1 - rain_mask)
                # sdr_loss = loss_map.sum() / ((1 - rain_mask).sum() + 1e-6)
                # # print(masked_loss, non_rain_loss)
                # T = 100
                # a = max(0.0, 1.0 - j / T)   # 1 → 0
                # loss = masked_loss + non_rain_loss
                # loss = (a + 0.2 * pixel_loss.mean()) + (1 - a) * new_loss
                # loss = ((a + 0.2) * pixel_loss.mean()) + (1 - a) * new_loss
                loss = masked_loss + non_rain_loss + (0.4 * uncertain_loss)
                # loss = pixel_loss.mean()
                # if (j+1) < 70:
                #     loss = pixel_loss.mean()
                # else:
                #     loss = masked_loss + non_rain_loss + (0.2 * sdr_loss)
                loss_epoch += loss
                # loss = masked_loss + non_rain_loss
                # loss = pixel_loss + self_consistency_loss
                # mask_consistency_pixel

                # pr_loss = LocalP50WeightedL1Loss()
                # ldgp_loss = LocalMeanWeightedLoss()
                # feature_loss = min_rgb_loss(input_img, net_output)
                # new_loss, weight_map = pr_loss(input_img, net_output, sdr_images)
                # loss = pixel_loss.mean() # 20251022_only_pixel_loss
                # loss = (0.5 * new_loss) + (feature_loss * (1 - weight_map)).mean() + (0.1 * pixel_loss.mean())
                # new_loss, weight_map = ldgp_loss(input_img, net_output, sdr_images)
                # self_loss, _ = ldgp_loss(input_img, net_output, input_img)
                # tv = tv_loss(net_output)
                # loss = new_loss + ((1 - weight_map) * feature_loss).mean()
                # loss = new_loss + (0.5 * pixel_loss.mean()) # 20251116 new loss 
                # loss = new_loss + self_loss + pixel_loss.mean() # 20251118
                # loss = new_loss + (0.5 * pixel_loss.mean()) # 20251119
                # loss = pixel_loss.mean() # original

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            # print(f"average loss = {loss_epoch / k}")
        
        # inference
        # if (j+1) % 10 == 0:
            model.eval()
            net_output = model(rainy_images)
            val_pixel_loss = ((net_output - sdr_images_target) ** 2)
            val_pixel_loss = val_pixel_loss.mean()
            val_loss_all = val_pixel_loss
            # if val_loss_all < best_val_loss or True:
            #     print("warning: now is the true mode")
            #     best_epoch = j
            #     # print(f"epoch {j}, best epoch update !")
            #     best_val_loss = val_loss_all
            #     net_output = net_output[:,:,:h,:w]
            #     denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
            #     # plt.imsave(os.path.join(save_path,name[0]), denoised)
            #     plt.imsave(os.path.join(save_path,"test_" + str(j) + ".png"), denoised)
            # elif (j+1) % 10 == 0:
            #     print("warning: now is the testing time")
            #     best_epoch = j
            #     # print(f"epoch {j}, best epoch update !")
            #     best_val_loss = val_loss_all
            #     net_output = net_output[:,:,:h,:w]
            #     denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
            #     # plt.imsave(os.path.join(save_path,name[0]), denoised)
            #     plt.imsave(os.path.join(save_path,"test_" + str(j) + ".png"), denoised)
        print(f"The best epoch is : {best_epoch}")

        net_output = net_output[:,:,:h,:w]
        denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        print("warning, now we don't use validation loss")

        plt.imsave(os.path.join(save_path,name[0]), denoised)

        time = epoch_timer.toc()
        print("Time: ", time)
        total_time += time
    
    except Exception as e:
        print("Exception occur: ", e)
        pass
    
print("Finish! Average Time:", total_time/len(data_loader))