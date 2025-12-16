import os
from PIL import Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import torch.nn.functional as NF

from augment import Compose, RandomCrop, RandomHorizontalFilp, ToTensor
import random
import time
import torch

g = torch.Generator()
g.manual_seed(int(time.time()))  # 用時間當作 seed

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
        label = Image.open(sdr_img_path).convert("RGB")
        
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
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class SDR_Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = os.listdir(os.path.join(image_dir)) 
        self._check_image(self.image_list)
        self.image_list.sort()
        self.crop_size = 256
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # sdr_img = Image.open(os.path.join(self.image_dir, self.image_list[idx])).convert("RGB")

        parent_dir = os.path.dirname(os.path.dirname(self.image_dir))
        last_dir = os.path.basename(self.image_dir) 

        base_img_path = os.path.join(parent_dir, "input")
        rain_mask_path = os.path.join(parent_dir, "ldgp")
        # non_rain_mask_path = os.path.join(parent_dir, "non_rain_mask")
        non_rain_mask_path = os.path.join(parent_dir, "ldgp")
        sdr_fuse_path = os.path.join(parent_dir, "sdr_fuse")
        new_sdr_path = os.path.join(parent_dir, "sdr_fuse") # now we don't use 
        if os.path.exists(os.path.join (base_img_path, (last_dir + ".png"))):
            input_img_path = os.path.join (base_img_path, (last_dir + ".png"))
            rain_mask_path = os.path.join (rain_mask_path, (last_dir + ".png"))
            non_rain_mask_path = os.path.join (non_rain_mask_path, (last_dir + ".png"))
            sdr_fuse_path = os.path.join (sdr_fuse_path, (last_dir + ".png"))
            new_sdr_path = os.path.join (new_sdr_path, (last_dir + ".png"))
        else:
            input_img_path = os.path.join (base_img_path, (last_dir + ".jpg"))
            rain_mask_path = os.path.join (rain_mask_path, (last_dir + ".jpg"))
            non_rain_mask_path = os.path.join (non_rain_mask_path, (last_dir + ".jpg"))
            sdr_fuse_path = os.path.join (sdr_fuse_path, (last_dir + ".jpg"))
            new_sdr_path = os.path.join (new_sdr_path, (last_dir + ".jpg"))


        sdr_edge_map_path = parent_dir + "/sdr_edge/" + last_dir
        sdr_edge_map_path = os.path.join(sdr_edge_map_path, self.image_list[idx])

        input_img = Image.open(input_img_path).convert("RGB")
        rain_mask = Image.open(rain_mask_path).convert("L")
        non_rain_mask = Image.open(non_rain_mask_path).convert("L")
        sdr_img = Image.open(sdr_fuse_path).convert("RGB")
        new_sdr_img = Image.open(new_sdr_path).convert("RGB")

        sdr_img = F.to_tensor(sdr_img)
        input_img = F.to_tensor(input_img)
        rain_mask = F.to_tensor(rain_mask)
        non_rain_mask = F.to_tensor(non_rain_mask)
        new_sdr_img = F.to_tensor(new_sdr_img)

        _, h, w = input_img.shape
        if h > self.crop_size and w > self.crop_size:
            # sdr_img, input_img, rain_mask = self.random_crop_pair(sdr_img, input_img, rain_mask, crop_size=self.crop_size)
            sdr_img, input_img, rain_mask, non_rain_mask, new_sdr_img = self.random_crop_pair(sdr_img, input_img, rain_mask, non_rain_mask, new_sdr_img, crop_size=self.crop_size)
            sdr_img, input_img, rain_mask, non_rain_mask, new_sdr_img = self.random_flip(sdr_img, input_img, rain_mask, non_rain_mask, new_sdr_img)
        else:
            factor = 16
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            sdr_img = NF.pad(sdr_img, (0,padw,0,padh), 'reflect')
            input_img = NF.pad(input_img, (0,padw,0,padh), 'reflect')
            rain_mask = NF.pad(rain_mask, (0,padw,0,padh), 'reflect')

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

def Segmentation_dataloader(path, batch_size=4, num_workers=0):
    dataloader = DataLoader(
        Segmentation_Dataset(path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class Segmentation_Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = os.listdir(os.path.join(image_dir)) 
        self._check_image(self.image_list)
        self.image_list.sort()
        self.crop_size = 256
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        sdr_img = Image.open(os.path.join(self.image_dir, self.image_list[idx])).convert("RGB")

        parent_dir = os.path.dirname(os.path.dirname(self.image_dir))
        last_dir = os.path.basename(self.image_dir) 

        base_img_path = os.path.join(parent_dir, "input")
        # rain_mask_dir_path = os.path.join(parent_dir, "rain_mask_difference")
        rain_mask_dir_path = os.path.join(parent_dir, "ldgp")
        if os.path.exists(os.path.join (base_img_path, (last_dir + ".png"))):
            input_img_path = os.path.join (base_img_path, (last_dir + ".png"))
            rain_mask_path = os.path.join (rain_mask_dir_path, (last_dir + ".png"))
        else:
            input_img_path = os.path.join (base_img_path, (last_dir + ".jpg"))
            rain_mask_path = os.path.join (rain_mask_dir_path, (last_dir + ".jpg"))

        input_img = Image.open(input_img_path).convert("RGB")
        rain_mask = Image.open(rain_mask_path).convert("L")

        sdr_img = F.to_tensor(sdr_img)
        input_img = F.to_tensor(input_img)
        rain_mask = F.to_tensor(rain_mask)

        _, h, w = input_img.shape
        if h > self.crop_size and w > self.crop_size:
            rain_mask, input_img = self.random_crop_pair(rain_mask, input_img, crop_size=self.crop_size)
            sdr_img, _ = self.random_crop_pair(sdr_img, input_img, crop_size=self.crop_size)
        else:
            factor = 16
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            rain_mask = NF.pad(rain_mask, (0,padw,0,padh), 'reflect')
            input_img = NF.pad(input_img, (0,padw,0,padh), 'reflect')
            sdr_img = NF.pad(sdr_img, (0,padw,0,padh), 'reflect')

        return rain_mask, input_img, sdr_img

    @staticmethod
    def random_crop_pair(t1, t2, crop_size=256):
        _, h, w = t1.shape
        ch, cw = crop_size, crop_size
        
        if h < ch or w < cw:
            raise ValueError(f"Image too small for crop: ({h}, {w}) vs ({ch}, {cw})")

        i = random.randint(0, h - ch)  # 高度起點
        j = random.randint(0, w - cw)  # 寬度起點

        return t1[:, i:i+ch, j:j+cw], t2[:, i:i+ch, j:j+cw]
    
    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError