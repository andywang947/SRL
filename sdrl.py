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
# from PANet_data import PANet_dataloader

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

        # ##### train the segmentation model
        # seg_model = Seg_UNet(is_target=True, output_channels=1)
        # seg_model = seg_model.to(device)
        # seg_optimizer = Adam(seg_model.parameters(), lr=0.001)
        # seg_model.train()
        # seg_loader = Segmentation_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=4)

        # for j in tqdm(range(epochs)):
        #     seg_model.train()

        #     for k, inner_batch in enumerate(seg_loader):
        #         rain_mask, input_img, sdr_img = inner_batch
                
        #         rain_mask = rain_mask.to(device)
        #         input_img = input_img.to(device)
        #         sdr_img = sdr_img.to(device)

        #         net_output = seg_model(input_img)
        #         sdr_net_output = seg_model(sdr_img)

        #         origin_loss = loss_function(net_output, rain_mask)
        #         zero_mask = torch.zeros_like(sdr_net_output)
        #         self_loss = loss_function(sdr_net_output, zero_mask)

        #         loss = origin_loss + self_loss

        #         seg_optimizer.zero_grad()
        #         loss.backward()
        #         seg_optimizer.step()
        
        # seg_model.eval()

        # rainy_images = rainy_images.to(device)
        # clean_images = clean_images.to(device)

        # input_mask = seg_model(rainy_images)
        # gt_mask = seg_model(clean_images)
        # input_mask = input_mask[0].squeeze(0).detach().cpu().numpy()  # shape [H,W]
        # input_mask = np.clip(input_mask, 0, 1)
        # plt.imsave(os.path.join(save_path,"test_input_mask.png"), input_mask, cmap="gray")

        # gt_mask = seg_model(clean_images)
        # gt_mask = gt_mask[0].squeeze(0).detach().cpu().numpy()  # shape [H,W]
        # gt_mask = np.clip(gt_mask, 0, 1)
        # plt.imsave(os.path.join(save_path,"test_gt_mask.png"), gt_mask, cmap="gray")

        # break
 
        # ##### ending the training of the segmentation model   

        # training for transfer rain

        # from PANet import PHATNet
        # transfer_model = PHATNet().to(device)
        # PHATNet_optimizer = Adam(transfer_model.parameters(), lr=0.001)

        # PHATNet_loader = PANet_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=8)
        # loss_function = torch.nn.L1Loss()
        # print("note: now is to use L1 loss for training.")
        # for j in tqdm(range(epochs)):
        #     transfer_model.train()
        #     for k, inner_batch in enumerate(PHATNet_loader):
        #         sdr_img, rainy_img, another_sdr_img = inner_batch
                
        #         sdr_img = sdr_img.to(device)
        #         rainy_img = rainy_img.to(device)
        #         another_sdr_img = another_sdr_img.to(device)

        #         rehaze_0, rehaze_1, rehaze_2 = transfer_model(rainy_img, sdr_img)
        #         reclean_0, reclean_1, reclean_2 = transfer_model(another_sdr_img, sdr_img)

        #         rainy_img = rainy_img[:, :3, :, :] # get out the rain mask channel

        #         rain_down_1 = F.interpolate(rainy_img, scale_factor=0.5, mode='bilinear')
        #         rain_down_2 = F.interpolate(rainy_img, scale_factor=0.25, mode='bilinear')

        #         loss_0 = loss_function(rehaze_0, rainy_img)
        #         # loss_1 = loss_function(rehaze_1, rain_down_1)
        #         # loss_2 = loss_function(rehaze_2, rain_down_2)
        #         # rain_loss = loss_0 + loss_1 + loss_2
        #         rain_loss = loss_0

        #         sdr_down_1 = F.interpolate(sdr_img, scale_factor=0.5, mode='bilinear')
        #         sdr_down_2 = F.interpolate(sdr_img, scale_factor=0.25, mode='bilinear')

        #         self_loss_0 = loss_function(reclean_0, sdr_img)
        #         # self_loss_1 = loss_function(reclean_1, sdr_down_1)
        #         # self_loss_2 = loss_function(reclean_2, sdr_down_2)
        #         self_loss = self_loss_0
        #         print(rain_loss.item(), self_loss.item())
        #         if k < 50:
        #             loss = rain_loss
        #         else:
        #             loss = rain_loss + self_loss
        #         # loss = rain_loss 
        #         # loss = loss_0 + self_loss_0

        #         PHATNet_optimizer.zero_grad()
        #         loss.backward()
        #         PHATNet_optimizer.step()
        
        #     # inference
        #     another_sdr_img = np.clip(another_sdr_img[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        #     reclean_0 = np.clip(reclean_0[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        #     rehaze_0 = np.clip(rehaze_0[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        #     plt.imsave("training_rerain.png", rehaze_0)
        #     plt.imsave("training_reclean.png", reclean_0)

        #     from torchvision.transforms import functional as Func
        #     ldgp_img_path = "dataset/test/ldgp/1.png"
        #     ldgp_img = Image.open(ldgp_img_path).convert("L")
        #     ldgp_img = Func.to_tensor(ldgp_img)
        #     ldgp_img = ldgp_img.unsqueeze(0)
        #     ldgp_img = ldgp_img.to(device)
        #     inference_rainy_images = rainy_images.to(device)

        #     inference_rainy_images = torch.cat([inference_rainy_images, ldgp_img], dim=1)  # shape [4, H, W]


        #     # rainy_images_vflip = torch.flip(inference_rainy_images, dims=[2])  # 如果維度是 (C, H, W)，這樣會上下翻轉
        #     # rainy_images_vflip, clean_images = rainy_images_vflip.to(device), clean_images.to(device) 
        #     transfer_model.eval()
        #     # rehaze_0, _, _= transfer_model(rainy_images_vflip, clean_images)
        #     clean_images = clean_images.to(device)
        #     rehaze_0, _, _= transfer_model(inference_rainy_images, clean_images)
        #     time = epoch_timer.toc()
        #     print("Time: ", time)
        #     total_time += time
        #     rehaze_0 = rehaze_0[:,:,:h,:w]

        #     denoised = np.clip(rehaze_0[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        #     plt.imsave("test.png", denoised)
        # exit()

        # #  
        
        model = UNet(is_target=True)

        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=0.001)

        inner_batch_size = 1
        rainy_images = rainy_images.to(device)
        sdr_images_target = sdr_images_target.to(device)
        ldgp_img = ldgp_img.to(device)

        model.train()

        SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=inner_batch_size)

        best_epoch = 0
        best_val_loss = float('inf')

        for j in tqdm(range(epochs)):
        # for j in range(epochs):
            model.train()

            for k, inner_batch in enumerate(SDR_loader):
                sdr_images, input_img, rain_mask = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)

                net_output = model(input_img)
                self_net_output = model(net_output)
                # rain_pred_net_output = seg_model(input_img)

                # sdr_loss = loss_function(net_output, sdr_images)
                pixel_loss = ((net_output - sdr_images) ** 2)
                masked_loss = (pixel_loss * rain_mask).mean()
                # loss = pixel_loss + self_consistency_loss
                # loss = masked_loss + self_consistency_loss + 0.5 * pixel_loss.mean()
                # loss = masked_loss + self_consistency_loss + 0.5 * pixel_loss.mean()
                # mask_consistency_pixel
                ##### channel consistency loss
                def channel_consistency_loss(net_output, input_img):
                    R_out, G_out, B_out = net_output[:, 0:1, :, :], net_output[:, 1:2, :, :], net_output[:, 2:3, :, :]
                    R_ori, G_ori, B_ori = input_img[:, 0:1, :, :], input_img[:, 1:2, :, :], input_img[:, 2:3, :, :]

                    # 計算各通道間的差
                    out_RG, out_GB, out_BR = R_out - G_out, G_out - B_out, B_out - R_out
                    ori_RG, ori_GB, ori_BR = R_ori - G_ori, G_ori - B_ori, B_ori - R_ori

                    # 平均三個通道的平方差
                    loss = (
                        torch.mean((out_RG - ori_RG) ** 2) +
                        torch.mean((out_GB - ori_GB) ** 2) +
                        torch.mean((out_BR - ori_BR) ** 2)
                    ) / 3.0

                    return loss
                color_consistency_loss = channel_consistency_loss(net_output, input_img)
                ######

                # loss = masked_loss + color_consistency_loss + pixel_loss.mean() # 20251020_best_val
                loss = pixel_loss.mean() # 20251022_only_pixel_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # inference
        # if (j+1) % 10 == 0:
            model.eval()
            net_output = model(rainy_images)
            val_pixel_loss = ((net_output - sdr_images_target) ** 2)
            val_masked_loss = (val_pixel_loss * ldgp_img).mean()
            val_pixel_loss = val_pixel_loss.mean()
            val_color_consistency_loss = channel_consistency_loss(net_output, rainy_images)
            val_loss_all = val_masked_loss + val_color_consistency_loss
            # val_loss_all = val_pixel_loss
            if val_loss_all < best_val_loss:
                best_epoch = j
                # print(f"epoch {j}, best epoch update !")
                best_val_loss = val_loss_all
                net_output = net_output[:,:,:h,:w]
                denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                # plt.imsave(os.path.join(save_path,name[0]), denoised)
                # plt.imsave(os.path.join(save_path,"test_" + str(j) + ".png"), denoised)
        print(f"The best epoch is : {best_epoch}")
        plt.imsave(os.path.join(save_path,name[0]), denoised)

        time = epoch_timer.toc()
        print("Time: ", time)
        total_time += time
    
    except Exception as e:
        print("Exception occur: ", e)
        pass
    
print("Finish! Average Time:", total_time/len(data_loader))