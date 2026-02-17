import torch
from torch.optim import Adam
from torch.nn import MSELoss
import torchvision.transforms as T

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from network import UNet
from data_all import train_all_dataloader
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./dataset/Rain100L/", help='path to input data')
parser.add_argument("--save_path", type=str, default="./result/Rain100L/", help='path to save result')

parser.add_argument("--seed", type=int, default=10, help='random seed')
parser.add_argument("--psd_num", type=int, default=50, help='the num of pseudo gt')
parser.add_argument("--mode", type=str, default="train_all", choices=['train_single', 'train_all', 'train_single_pyramid'],help='training mode')

# train_single parameter
parser.add_argument("--single_epoch", type=int, default=1, help='training epoch')
parser.add_argument("--f1", type=int, default=1, help='training epoch')
parser.add_argument("--f2", type=int, default=1, help='training epoch')

# train_all parameter
parser.add_argument("--epoch", type=int, default=2000, help='training epoch')
parser.add_argument("--batch_size", type=int, default=64, help='training batch size')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr_steps', type=list, default=[(x+1) * 100 for x in range(1000//100)])
parser.add_argument('--model_save_dir', type=str, default="./result_all/Rain100L/")

opt = parser.parse_args()

loss_function = MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_all(opt):
   writer = SummaryWriter()
   random.seed(opt.seed)
   data_loader = train_all_dataloader(opt.data_path, batch_size=opt.batch_size)
   model = UNet().to(device)
   optimizer = Adam(model.parameters(), lr=0.001)
   scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_steps, opt.gamma)

   for epoch in tqdm(range(opt.epoch)):
      for itx, batch in enumerate(data_loader):
         rainy_images, clean_images, ldgp_mask, sdr_images, name = batch
         rainy_images = rainy_images.to(device)
         clean_images = clean_images.to(device)
         ldgp_mask    = ldgp_mask.to(device)
         sdr_images   = sdr_images.to(device)
         _, _, H, W = rainy_images.shape
         
         more_rainy_images = rainy_images

         net_output = model(more_rainy_images)
         loss = loss_function(net_output, sdr_images)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
   
      if epoch%10==0:
         print("Epoch: ", epoch, "Epoch Loss: ", loss.item())
   
   model_name = os.path.join(opt.model_save_dir, 'model.pkl')
   torch.save({'model': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict()}, model_name)

train_all(opt)