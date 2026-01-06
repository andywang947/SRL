import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

from torch.optim import Adam
from tqdm import tqdm

from utils import Timer
from network import UNet
from data import SDR_dataloader, train_dataloader
from itertools import islice
import torch.nn.functional as F
from mask_aug import shuffle_connected_components_torch

torch.manual_seed(3)
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="test", help="Dataset name")
parser.add_argument("--result_name", type=str, default="test", help="Dataset name")
parser.add_argument("--epoch", type=int, default=100)

opt = parser.parse_args()

opt.rainy_data_path = f"./dataset/{opt.dataset}/"
opt.sdr_data_path = f"./dataset/{opt.dataset}/sdr/"
opt.result_path = f"./dataset/{opt.dataset}/result_{opt.result_name}/"

data_path = opt.rainy_data_path
save_path = opt.result_path
sdr_path = opt.sdr_data_path
epochs = opt.epoch

data_loader = train_dataloader(data_path, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(save_path, exist_ok=True)

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

        addrain_model = UNet(input_channels=4, output_channels=3, is_target=True)
        addrain_model = addrain_model.to(device)

        # from network import UNetDiscriminatorSN
        # d_model = UNetDiscriminatorSN(num_in_ch=4)

        # from diffusion_model import UNet_addrain
        # from diffusion_network import DDIM
        # addrain_unet = UNet_addrain(img_channels=7,dropout=0.0).to(device)
        # def generate_linear_schedule(T, beta_1, beta_T):
        #     return torch.linspace(beta_1, beta_T, T).double()
        # beta = generate_linear_schedule(2000, 1e-6, 1e-2)
        # addrain_model = DDIM(addrain_unet, img_channels=7, betas=beta, criterion="l1").to(device)

        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        addrain_optimizer = Adam(addrain_model.parameters(), lr=0.001)
        addrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(addrain_optimizer, T_max=epochs)

        inner_batch_size = 1
        rainy_images = rainy_images.to(device)
        sdr_images_target = sdr_images_target.to(device)
        ldgp_img = ldgp_img.to(device)

        model.train()
        addrain_model.train()
        SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=inner_batch_size)

        for epoch in tqdm(range(epochs)):
            addrain_model.train()
            for k, inner_batch in enumerate(islice(SDR_loader, 200)):
                sdr_images, input_img, rain_mask, _, _ = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)

                addrain_optimizer.zero_grad()
                # rain_mask = binary_mask_to_soft(rain_mask)
                addrain_input = torch.cat([sdr_images, rain_mask],dim=1) # standard, use sdr to addrain to input
                # addrain_input = torch.cat([sdr_images, rain_mask],dim=1) # standard, use sdr to addrain to input
                addrain_output = addrain_model(addrain_input)

                # loss = addrain_model(x=input_img, condition=addrain_input)

                loss = torch.abs(addrain_output - input_img).mean()

                loss.backward()
                addrain_optimizer.step()

            addrain_scheduler.step()
        
            # inference
            addrain_model.eval()
            with torch.no_grad():
                if opt.result_name == "test":
                    addrain_mask = torch.flip(ldgp_img, dims=[2, 3])
                    # addrain_mask_weight = binary_mask_to_soft(ldgp_img)
                    addrain_mask = shuffle_connected_components_torch(ldgp_img)
                    addrain_input = torch.cat([rainy_images, addrain_mask],dim=1)
                    # addrain_input = torch.cat([sdr_images_target, addrain_mask_weight],dim=1)
                    net_output = addrain_model(addrain_input)

                    # net_output = addrain_model.sample(condition=addrain_input,sample_timesteps=10, device=device)
                    # net_output = net_output.to(device)

                    net_output = net_output[:,:,:h,:w]
                    # denoised = addrain_mask[0, 0].detach().cpu().numpy()
                    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    plt.imsave(os.path.join(save_path,name[0]), denoised)

        skip_batch = False
        for j in tqdm(range(epochs)):
            if os.path.exists(img_save_path) == True and opt.result_name != "test":
                skip_batch = True
                break
            model.train()
            addrain_model.eval()

            for k, inner_batch in enumerate(islice(SDR_loader, 50)):
                sdr_images, input_img, rain_mask, non_rain_mask, gt = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)
                non_rain_mask = non_rain_mask.to(device)

                with torch.no_grad():
                    # non_rain_mask = shuffle_connected_components_torch(non_rain_mask)
                    addrain_input = torch.cat([input_img, non_rain_mask],dim=1)
                    addrain_input = addrain_model(addrain_input)

                net_output = model(addrain_input)
                ori_loss = torch.abs(net_output - sdr_images).mean()
                loss = ori_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
        
            # inference
            model.eval()
            with torch.no_grad():
                if opt.result_name == "test":
                    net_output = model(rainy_images)
                    net_output = net_output[:,:,:h,:w]
                    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    plt.imsave(os.path.join(save_path,name[0]), denoised)
                    # plt.imsave(os.path.join(save_path,"test_" + str(j) + ".png"), denoised)
                        # plt.imsave(os.path.join(save_path,"test_" + str(i) + ".png"), denoised)

        if skip_batch:
            print("in another process, the training for this image is done.")
            continue
        with torch.no_grad():
            net_output = model(rainy_images)
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