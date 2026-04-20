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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="./dataset/Rain100L_train/", help='path to input data')

parser.add_argument("--seed", type=int, default=10, help='random seed')

# train_all parameter
parser.add_argument("--epoch", type=int, default=500, help='training epoch')
parser.add_argument("--batch_size", type=int, default=64, help='training batch size')
parser.add_argument("--lr_rate", type=float, default=0.001, help='training batch size')
parser.add_argument('--model_save_dir', type=str, default="./result_all/weight/")
parser.add_argument('--result_name', type=str, default="test")
parser.add_argument("--pretrain",action="store_true",help="enable pretraining mode")


opt = parser.parse_args()

os.makedirs(opt.model_save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_loader = train_all_dataloader(opt.dataset, batch_size=opt.batch_size)
model = UNet().to(device)

if opt.pretrain:
   pretrained_weight_path = "result_all/Rain100L/only_pseudo.pkl"
   print(f"[warning]: now use the pretrained weight. from {pretrained_weight_path}")
   model_state_dict = torch.load(pretrained_weight_path)
   model.load_state_dict(model_state_dict['model'])

optimizer = Adam(model.parameters(), lr=opt.lr_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)

for epoch in tqdm(range(opt.epoch)):
   model.train()
   epoch_loss = 0.0
   for itx, batch in enumerate(data_loader):
      rainy_images, clean_images, name = batch
      rainy_images = rainy_images.to(device)
      clean_images = clean_images.to(device)
      _, _, H, W = rainy_images.shape
      
      net_output = model(rainy_images)
      loss = torch.abs(net_output - clean_images).mean()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
   
   scheduler.step()
   
   epoch_loss /= len(data_loader)
   print("Epoch: ", epoch, "Epoch Loss: ", epoch_loss)

   if (epoch + 1) % 50 == 0:
      print("save weight")
      model_name = os.path.join(opt.model_save_dir, (opt.result_name + str(epoch+1) + '.pkl'))
      torch.save({'model': model.state_dict()}, model_name)