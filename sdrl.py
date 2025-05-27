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
from network import UNet, ResNet, DnCNN
from restormer import Restormer
from data import SDR_dataloader, train_dataloader

import torch.nn.functional as F


torch.manual_seed(3)
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="DDN_SIRR_real", help="Dataset name")
parser.add_argument("--result_name", type=str, default="result_reparameter_one_layer_20250527", help="Dataset name")
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
        rainy_images, clean_images, name = batch

        h,w = rainy_images.shape[2], rainy_images.shape[3]
        factor = 16

        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        rainy_images = F.pad(rainy_images, (0,padw,0,padh), 'reflect')

        img_save_path = os.path.join(save_path,name[0])
        print(img_save_path)
        if os.path.exists(img_save_path) == True :
            print("the image exists!")
            continue
        else :
            print("The image now is :", name[0])

        epoch_timer.tic()    
        
        if opt.backbone == "Unet":
            model = UNet(is_target=True)
            aux_model = UNet(input_channels=1)

        elif opt.backbone == "ResNet":
            model = ResNet()
        elif opt.backbone == "DnCNN":
            model = DnCNN()
        elif opt.backbone == "Restormer":
            model = Restormer()

        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=0.001)

        aux_model = aux_model.to(device)
        aux_optimizer = Adam(aux_model.parameters(), lr=0.001)

        inner_batch_size = 1
        rainy_images = rainy_images.to(device)
        clean_images = clean_images.to(device)

        model.train()
        aux_model.train()

        SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=inner_batch_size)
        

        for j in tqdm(range(epochs)):
            model.train()

            for k, inner_batch in enumerate(SDR_loader):
                sdr_images, input_edge_map, sdr_edge_map = inner_batch

                sdr_images = F.pad(sdr_images, (0,padw,0,padh), 'reflect')
                input_edge_map = F.pad(input_edge_map, (0,padw,0,padh), 'reflect')
                sdr_edge_map = F.pad(sdr_edge_map, (0,padw,0,padh), 'reflect')

                sdr_images = sdr_images.to(device)
                input_edge_map = input_edge_map.to(device)
                sdr_edge_map = sdr_edge_map.to(device)

                images = torch.cat([rainy_images for _ in range(len(sdr_images))],0)

                aux_net_output = aux_model(input_edge_map)
                aux_loss = loss_function(aux_net_output, sdr_edge_map)
                aux_optimizer.zero_grad()
                aux_loss.backward()
                aux_optimizer.step()

                net_output = model(images, aux_model=aux_model)
                loss = loss_function(net_output, sdr_images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # inference

        model.eval()
        # net_output = model(rainy_images)
        net_output = model(rainy_images, aux_model=aux_model)
        time = epoch_timer.toc()
        print("Time: ", time)
        total_time += time
        net_output = net_output[:,:,:h,:w]

        denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        plt.imsave(os.path.join(save_path,name[0]), denoised)
        # plt.imsave(os.path.join(save_path,"test" + str(j) + ".png"), denoised)
    
    except Exception as e:
        print("Exception occur: ", e)
        pass
    
print("Finish! Average Time:", total_time/len(data_loader))