import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import torch.nn.functional as NF

from augment import Compose, RandomCrop, RandomHorizontalFilp, ToTensor
import random
import time
import torch

g = torch.Generator()
g.manual_seed(int(time.time()))

def train_dataloader(image_dir, batch_size=64, num_workers=0):
    transform = None
    dataloader = DataLoader(
        RainDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g
    )
    return dataloader


def test_dataloader(image_dir, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        RainDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

class RainDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.image_list = os.listdir(os.path.join(image_dir, 'input/')) 
        self._check_image(self.image_list)
        self.label_list = list()
        for i in range(len(self.image_list)):
            filename = self.image_list[i]
            self.label_list.append(filename)
        self.image_list.sort()
        self.label_list.sort()
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx])).convert("RGB")
        ldgp_img = Image.open(os.path.join(self.image_dir, 'ldgp', self.image_list[idx])).convert("L")
        sdr_img_path = os.path.join(self.image_dir, 'sdr', self.label_list[idx])
        sdr_img_path = sdr_img_path.replace(".png", "")
        sdr_img_path = sdr_img_path.replace(".jpg", "")
        sdr_img_path = os.path.join(sdr_img_path,"0.png")
        label = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx])).convert("RGB")
        
        image = F.to_tensor(image)
        label = F.to_tensor(label)
        ldgp_img = F.to_tensor(ldgp_img)
        
        name = self.image_list[idx]
        return image, label, name, ldgp_img
        
    
    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


def SDR_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        SDR_Dataset(path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader


class SDR_Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = os.listdir(os.path.join(image_dir)) 
        self._check_image(self.image_list)
        self.image_list.sort()
        self.crop_size = 64
        self.parent_dir = os.path.dirname(os.path.dirname(self.image_dir))
        self.input_img_path = os.path.join(self.parent_dir, "input")
        self.img_name = os.path.basename(self.image_dir) 
        # sdr_img = Image.open(os.path.join(self.image_dir, self.image_list[idx])).convert("RGB")

        rain_mask_path = os.path.join(self.parent_dir, "ldgp")
        # non_rain_mask_path = os.path.join(self.parent_dir, "nonrain_mean")
        non_rain_mask_path = os.path.join(self.parent_dir, "ldgp")
        sdr_fuse_path = os.path.join(self.parent_dir, "sdr_fuse")
        new_sdr_path = os.path.join(self.parent_dir, "sdr_fuse") # now we don't use 
        if os.path.exists(os.path.join (self.input_img_path, (self.img_name + ".png"))):
            input_img_path = os.path.join (self.input_img_path, (self.img_name + ".png"))
            rain_mask_path = os.path.join (rain_mask_path, (self.img_name + ".png"))
            non_rain_mask_path = os.path.join (non_rain_mask_path, (self.img_name + ".png"))
        else:
            input_img_path = os.path.join (self.input_img_path, (self.img_name + ".jpg"))
            rain_mask_path = os.path.join (rain_mask_path, (self.img_name + ".jpg"))
            non_rain_mask_path = os.path.join (non_rain_mask_path, (self.img_name + ".jpg"))
            
        sdr_fuse_path = os.path.join (sdr_fuse_path, (self.img_name + ".png"))
        new_sdr_path = os.path.join (new_sdr_path, (self.img_name + ".png"))

        self.input_img = Image.open(input_img_path).convert("RGB")
        self.rain_mask = Image.open(rain_mask_path).convert("L")
        self.non_rain_mask = Image.open(non_rain_mask_path).convert("L")
        self.sdr_img = Image.open(sdr_fuse_path).convert("RGB")
        self.new_sdr_img = Image.open(new_sdr_path).convert("RGB")

        self.sdr_img = F.to_tensor(self.sdr_img)
        self.input_img = F.to_tensor(self.input_img)
        self.rain_mask = F.to_tensor(self.rain_mask)
        self.non_rain_mask = F.to_tensor(self.non_rain_mask)
        self.new_sdr_img = F.to_tensor(self.new_sdr_img)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):

        sdr_img = self.sdr_img
        input_img = self.input_img
        rain_mask = self.rain_mask
        non_rain_mask = self.non_rain_mask
        new_sdr_img = self.new_sdr_img

        _, h, w = input_img.shape
        if h > self.crop_size and w > self.crop_size:
            # sdr_img, input_img, rain_mask = self.random_crop_pair(sdr_img, input_img, rain_mask, crop_size=self.crop_size)
            sdr_img, input_img, rain_mask, _, new_sdr_img = self.random_crop_pair(self.sdr_img, self.input_img, self.rain_mask, self.non_rain_mask, self.new_sdr_img, crop_size=self.crop_size)
            _, _, non_rain_mask, _, new_sdr_img = self.random_crop_pair(self.sdr_img, self.input_img, self.rain_mask, self.non_rain_mask, self.new_sdr_img, crop_size=self.crop_size)
            # _, _, non_rain_mask_2, _, _ = self.random_crop_pair(self.sdr_img, self.input_img, self.rain_mask, self.non_rain_mask, self.new_sdr_img, crop_size=self.crop_size)
            sdr_img, input_img, rain_mask, non_rain_mask, new_sdr_img = self.random_flip(sdr_img, input_img, rain_mask, non_rain_mask, new_sdr_img)
        else:
            factor = 16
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            sdr_img = NF.pad(sdr_img, (0,padw,0,padh), 'reflect')
            input_img = NF.pad(input_img, (0,padw,0,padh), 'reflect')
            rain_mask = NF.pad(rain_mask, (0,padw,0,padh), 'reflect')
            non_rain_mask = NF.pad(rain_mask, (0,padw,0,padh), 'reflect')
            non_rain_mask_2 = NF.pad(rain_mask, (0,padw,0,padh), 'reflect')

        return sdr_img, input_img, rain_mask, non_rain_mask, new_sdr_img

    @staticmethod
    def random_crop_pair(t1, t2, rain_mask, non_rain_mask, new_sdr_img, crop_size=256):
        _, h, w = t1.shape
        ch, cw = crop_size, crop_size
        
        if h < ch or w < cw:
            raise ValueError(f"Image too small for crop: ({h}, {w}) vs ({ch}, {cw})")

        i = random.randint(0, h - ch)  # 高度起點
        j = random.randint(0, w - cw)  # 寬度起點

        return t1[:, i:i+ch, j:j+cw], t2[:, i:i+ch, j:j+cw], rain_mask[:, i:i+ch, j:j+cw], non_rain_mask[:, i:i+ch, j:j+cw], new_sdr_img[:, i:i+ch, j:j+cw]
    

    @staticmethod
    def random_flip(*tensors):
        """
        tensors: list of [C, H, W]
        """
        # horizontal flip
        if random.random() < 0.5:
            tensors = [torch.flip(t, dims=[2]) for t in tensors]  # W

        # vertical flip
        if random.random() < 0.5:
            tensors = [torch.flip(t, dims=[1]) for t in tensors]  # H

        return tensors
    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError