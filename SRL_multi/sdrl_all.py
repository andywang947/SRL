import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

from torch.optim import Adam
from tqdm import tqdm

from utils import Timer
from network import UNet
from data_multi import SDR_dataloader, train_dataloader, Addrain_dataloader
from itertools import islice
import torch.nn.functional as F
from mask_aug import shuffle_connected_components_torch, shuffle_connected_components_torch_preserve_gray
import yaml
import time

torch.manual_seed(3)
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="test", help="Dataset name")
parser.add_argument("--result_name", type=str, default="test", help="Dataset name")
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--config",type=str, default="configs/config.yaml",help="Path to config file")

opt = parser.parse_args()

with open(opt.config, "r") as f:
    cfg = yaml.safe_load(f)

if not cfg["loss"]["stage_2"]["Region_of_L_RR_use_non_rain"]:
    print("[warning]: now L_RR uses full image, it's ablation for stage 2")
if not cfg["loss"]["stage_2"]["use_consistency"]:
    print("[warning]: didn't use consistency loss in stage 2, it's ablation")

opt.rainy_data_path = f"./dataset/{opt.dataset}/"
opt.sdr_data_path = f"./dataset/{opt.dataset}/sdr/"
opt.result_path = f"./dataset/{opt.dataset}/result_{opt.result_name}/"

data_path = opt.rainy_data_path
save_path = opt.result_path
sdr_path = opt.sdr_data_path
epochs = opt.epoch

data_loader = train_dataloader(data_path, batch_size=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(save_path, exist_ok=True)

epoch_timer = Timer('s') 
total_time = 0

img_save_path = os.path.join(save_path)
print(img_save_path)

        
model = UNet()
model = model.to(device)
        
from addrain_network import AddRainNet, AddRainNet_test
addrain_model = AddRainNet(input_channels=4, output_channels=3)
addrain_model = addrain_model.to(device)

addrain_optimizer = Adam(addrain_model.parameters(), lr=0.001)
addrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(addrain_optimizer, T_max=epochs)
addrain_model.train()

initial_model = UNet()
initial_model = initial_model.to(device)


optimizer = Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

pseudo_optimizer = Adam(initial_model.parameters(), lr=0.001)
pseudo_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pseudo_optimizer, T_max=epochs)

stage_1_start = time.time()

for epoch in tqdm(range(epochs)):
    addrain_model.train()
    for batch in data_loader:
        sdr_img, input_img, rain_mask, _, _, _, _ = batch

        sdr_img = sdr_img.to(device)
        input_img = input_img.to(device)
        rain_mask = rain_mask.to(device)

        addrain_optimizer.zero_grad()

        addrain_output = addrain_model(sdr_img, rain_mask, input_img)                
        loss = torch.abs(addrain_output - input_img).mean()
        loss.backward()
        addrain_optimizer.step()
    addrain_scheduler.step()

torch.save(addrain_model.state_dict(), os.path.join(save_path, "addrain.pth"))

addrain_model.eval()

with torch.no_grad():
    for batch in data_loader:
        sdr_img, input_img, rain_mask, _, _, _, _ = batch

        sdr_img = sdr_img.to(device)
        input_img = input_img.to(device)
        rain_mask = rain_mask.to(device)

        output = addrain_model(sdr_img, rain_mask, input_img)

        # 取第一張
        out = output[0].permute(1, 2, 0).cpu().numpy()
        out = np.clip(out, 0, 1)

        plt.imsave(os.path.join(save_path, "test.png"), out)
        break

stage_1_end = time.time()
stage_1_duration = stage_1_end - stage_1_start
print("Total time of stage 1: ", stage_1_duration)

stage_2_start = time.time()

for epoch in tqdm(range(epochs)):
    initial_model.train()
    for batch in data_loader:
        sdr_img, input_img, rain_mask, another_mask_1, _, _, _ = batch
        sdr_img = sdr_img.to(device)
        input_img = input_img.to(device)
        rain_mask = rain_mask.to(device)
        another_mask_1 = another_mask_1.to(device)

        with torch.no_grad():
            addrain_input = addrain_model(input_img, another_mask_1, input_img)
            another_mask_2 = shuffle_connected_components_torch(another_mask_1)
            addrain_input_2 = addrain_model(input_img, another_mask_2, input_img)

        pseudo_net_output_1 = initial_model(addrain_input)
        pseudo_net_output_2 = initial_model(addrain_input_2)

        reconstruction_loss_1 = ((torch.abs(pseudo_net_output_1 - input_img)) * (1 - rain_mask)).mean()
        reconstruction_loss_2 = ((torch.abs(pseudo_net_output_2 - input_img)) * (1 - rain_mask)).mean()

        reconstruction_loss = 0.5 * (reconstruction_loss_1 + reconstruction_loss_2)
        consistency_loss = torch.abs(pseudo_net_output_1 - pseudo_net_output_2).mean()

        loss = reconstruction_loss + consistency_loss
        pseudo_optimizer.zero_grad()
        loss.backward()
        pseudo_optimizer.step()

    pseudo_scheduler.step()

# inference
torch.save(initial_model.state_dict(), os.path.join(save_path, "initial_derainer.pth"))

initial_model.eval()

with torch.no_grad():
    for batch in data_loader:
        sdr_img, input_img, rain_mask, _, _, _, _ = batch

        sdr_img = sdr_img.to(device)
        input_img = input_img.to(device)
        rain_mask = rain_mask.to(device)

        output = initial_model(input_img)

        # 取第一張
        out = output[0].permute(1, 2, 0).cpu().numpy()
        out = np.clip(out, 0, 1)

        plt.imsave(os.path.join(save_path, "test_initial_derainer.png"), out)
        break

stage_2_end = time.time()
stage_2_duration = stage_2_end - stage_2_start
print("Total time of stage 2: ", stage_2_duration)

stage_3_start = time.time()

for epoch in tqdm(range(epochs)):
    model.train()

    initial_model.eval()
    addrain_model.eval()
    for batch in data_loader:
        _, input_img, _, another_mask_1, _, _, _ = batch
        input_img = input_img.to(device)
        another_mask_1 = another_mask_1.to(device)

        with torch.no_grad():
            addrain_input = addrain_model(input_img, another_mask_1, input_img)
            pseudo_net_output = initial_model(input_img)

        net_output = model(addrain_input)

        loss = torch.abs(net_output - pseudo_net_output).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

# inference
model.eval()
torch.save(model.state_dict(), os.path.join(save_path, "refined_derainer.pth"))


stage_3_end = time.time()
stage_3_duration = stage_3_end - stage_3_start
print("Total time of stage 3: ", stage_3_duration)

with torch.no_grad():
    for batch in data_loader:
        sdr_img, input_img, rain_mask, _, _, _, _ = batch

        sdr_img = sdr_img.to(device)
        input_img = input_img.to(device)
        rain_mask = rain_mask.to(device)

        output = model(input_img)

        # 取第一張
        out = output[0].permute(1, 2, 0).cpu().numpy()
        out = np.clip(out, 0, 1)

        plt.imsave(os.path.join(save_path, "test_refined_derainer.png"), out)
        break

print("Finish!")