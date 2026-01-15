import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

from torch.optim import Adam
from tqdm import tqdm

from utils import Timer
from network import UNet
from data import SDR_dataloader, train_dataloader, Addrain_dataloader
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
        
        model = UNet()
        model = model.to(device)

        pseudo_model = UNet()
        pseudo_model = pseudo_model.to(device)
        
        from addrain_network import AddRainNet, AddRainNet_test
        if opt.result_name == "test":
            print("warning: now is the test mode")
            addrain_model = AddRainNet_test(input_channels=4, output_channels=3)        
        else:
            addrain_model = AddRainNet(input_channels=4, output_channels=3)
        addrain_model = addrain_model.to(device)

        # mask_model = UNet(input_channels=3, output_channels=1)
        # mask_model = mask_model.to(device)

        # from network import UNetDiscriminatorSN
        # d_model = UNetDiscriminatorSN(num_in_ch=4).to(device)

        addrain_optimizer = Adam(addrain_model.parameters(), lr=0.001)
        addrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(addrain_optimizer, T_max=epochs)
        addrain_model.train()


        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # d_optimizer = Adam(d_model.parameters(), lr=0.001)
        # d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=epochs)

        pseudo_optimizer = Adam(pseudo_model.parameters(), lr=0.001)
        pseudo_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pseudo_optimizer, T_max=epochs)


        # mask_optimizer = Adam(mask_model.parameters(), lr=0.001)
        # mask_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mask_optimizer, T_max=epochs)

        inner_batch_size = 1
        rainy_images = rainy_images.to(device)
        sdr_images_target = sdr_images_target.to(device)
        ldgp_img = ldgp_img.to(device)

        model.train()
        Addrain_loader = Addrain_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=inner_batch_size)
        for epoch in tqdm(range(epochs)):
            addrain_model.train()
            # d_model.train()
            for k, inner_batch in enumerate(islice(Addrain_loader, 50)):
                sdr_images, input_img, rain_mask, another_rain_mask, another_input_img = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)
                another_rain_mask = another_rain_mask.to(device)
                another_input_img = another_input_img.to(device)

                addrain_optimizer.zero_grad()

                addrain_output_1 = addrain_model(sdr_images, rain_mask)                
                loss = torch.abs(addrain_output_1 - input_img).mean()

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

                    # addrain_input = torch.cat([rainy_images, addrain_mask],dim=1)
                    net_output = addrain_model(rainy_images, addrain_mask)

                    # net_output = addrain_model.sample(condition=addrain_input,sample_timesteps=10, device=device)
                    # net_output = net_output.to(device)

                    net_output = net_output[:,:,:h,:w]
                    # denoised = addrain_mask[0, 0].detach().cpu().numpy()
                    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    plt.imsave(os.path.join(save_path,name[0]), denoised)

        # print("[warning] dev for addrain model, end !")
        # exit()


        # ###### rain mask model
        # for epoch in tqdm(range(epochs)):
        #     mask_model.train()
        #     for k, inner_batch in enumerate(islice(SDR_loader, 200)):
        #         sdr_images, input_img, rain_mask, non_rain_mask, _ = inner_batch
                
        #         sdr_images = sdr_images.to(device)
        #         input_img = input_img.to(device)
        #         rain_mask = rain_mask.to(device)
        #         non_rain_mask = non_rain_mask.to(device)

        #         with torch.no_grad():
        #             # non_rain_mask = shuffle_connected_components_torch(non_rain_mask)
        #             addrain_input = torch.cat([input_img, non_rain_mask],dim=1)
        #             addrain_input = addrain_model(addrain_input)
        #             rain_mask_gt = torch.max(non_rain_mask, rain_mask)

        #         mask_optimizer.zero_grad()
        #         mask_input = addrain_input
        #         mask_output = mask_model(mask_input)

        #         loss = torch.abs(mask_output - rain_mask_gt).mean()

        #         loss.backward()
        #         mask_optimizer.step()

        #     mask_scheduler.step()
        
        #     # inference
        #     mask_model.eval()
        #     with torch.no_grad():
        #         if opt.result_name == "test":
        #             net_output = mask_model(rainy_images)
        #             net_output = (net_output > 0.5).float()
        #             net_output = net_output[:,:,:h,:w]
        #             # denoised = addrain_mask[0, 0].detach().cpu().numpy()
        #             # denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        #             denoised = np.clip(net_output[0, 0].detach().cpu().numpy(), 0, 1)
        #             plt.imsave(os.path.join(save_path,name[0]), denoised)
        #     ####### mask model end
        skip_batch = False
        SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=inner_batch_size)
        for j in tqdm(range(epochs)):
            if os.path.exists(img_save_path) == True and opt.result_name != "test":
                skip_batch = True
                break
            pseudo_model.train()

            for k, inner_batch in enumerate(islice(SDR_loader, 50)):
                sdr_images, input_img, rain_mask, non_rain_mask, another_input_img = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)
                non_rain_mask = non_rain_mask.to(device)
                with torch.no_grad():
                    addrain_mask_1 = non_rain_mask
                    addrain_mask_2 = shuffle_connected_components_torch(non_rain_mask)

                    addrain_input = addrain_model(input_img, addrain_mask_1)
                    addrain_input_2 = addrain_model(input_img, addrain_mask_2)

                pseudo_net_output_1 = pseudo_model(addrain_input)
                pseudo_net_output_2 = pseudo_model(addrain_input_2)
                # from loss import masked_tv_loss
                # tv_loss = masked_tv_loss(pseudo_net_output_1, rain_mask)

                self_loss_1 = ((torch.abs(pseudo_net_output_1 - input_img)) * (1 - rain_mask)).mean()
                self_loss_2 = ((torch.abs(pseudo_net_output_2 - input_img)) * (1 - rain_mask)).mean()
                self_loss = self_loss_1 + self_loss_2

                consistency_loss = torch.abs(pseudo_net_output_1 - pseudo_net_output_2).mean()
                # loss = self_loss + tv_loss + consistency_loss
                loss = (0.5 * self_loss) + consistency_loss
                pseudo_optimizer.zero_grad()
                loss.backward()
                pseudo_optimizer.step()

            pseudo_scheduler.step()

        skip_batch = False
        for j in tqdm(range(epochs)):
            if os.path.exists(img_save_path) == True and opt.result_name != "test":
                skip_batch = True
                break
            model.train()
            pseudo_model.train()
            addrain_model.eval()

            for k, inner_batch in enumerate(islice(SDR_loader, 50)):
                sdr_images, input_img, rain_mask, non_rain_mask, gt = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)
                non_rain_mask = non_rain_mask.to(device)

                # ##### training for the addrain model
                # addrain_model.train()
                # addrain_optimizer.zero_grad()

                # addrain_input = torch.cat([sdr_images, rain_mask],dim=1) # standard, use sdr to addrain to input
                # addrain_output = addrain_model(addrain_input)
                # loss = torch.abs(addrain_output - input_img).mean()

                # loss.backward()
                # addrain_optimizer.step()
                # addrain_model.eval()

                # ##### end of training for the addrain model

                with torch.no_grad():

                    # addrain_mask_1 = torch.max(non_rain_mask, rain_mask)
                    addrain_mask_1 = non_rain_mask
                    addrain_input = addrain_model(input_img, addrain_mask_1)
                    addrain_mask_2 = shuffle_connected_components_torch(non_rain_mask)
                    addrain_input_2 = addrain_model(input_img, addrain_mask_2)

                #     addrain_mask_3 = shuffle_connected_components_torch(non_rain_mask)
                #     addrain_mask_3 = torch.max(addrain_mask_3, rain_mask)
                #     addrain_input_3 = torch.cat([input_img, addrain_mask_3],dim=1)
                #     addrain_input_3 = addrain_model(addrain_input_3)
                pseudo_net_output = pseudo_model(addrain_input)

                # net_output_2 = model(addrain_input_2)
                # net_output_3 = model(addrain_input_3)
    
                # # ori_loss= (torch.abs(net_output - sdr_images)).mean()
                # or cosine:
                # alpha = 0.5 * (1 + math.cos(math.pi * epoch / T))
                with torch.no_grad():
                    pseudo_net_output = pseudo_model(input_img)
                net_output = model(addrain_input)
                net_output_2 = model(addrain_input_2)

                pseudo_label = input_img * (1 - rain_mask) + pseudo_net_output * (rain_mask)
                self_loss = torch.abs(net_output - pseudo_label).mean()

                consistency_loss = torch.abs(net_output - net_output_2) * (1 - rain_mask)
                consistency_loss = consistency_loss.mean()

                loss = self_loss + consistency_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # addrain_scheduler.step()
            # pseudo_scheduler.step()
            scheduler.step()
        
            # inference
            model.eval()
            with torch.no_grad():
                if opt.result_name == "test":
                    # mask_feature = mask_model(rainy_images, return_bottleneck=True)
                    net_output = model(rainy_images)
                    # net_output = model(rainy_images)
                    net_output = net_output[:,:,:h,:w]
                    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    plt.imsave(os.path.join(save_path,name[0]), denoised)
                    # plt.imsave(os.path.join(save_path,"test_" + str(j) + ".png"), denoised)
                        # plt.imsave(os.path.join(save_path,"test_" + str(i) + ".png"), denoised)
                    addrain_mask = torch.flip(ldgp_img, dims=[2, 3])
                    # addrain_mask_weight = binary_mask_to_soft(ldgp_img)
                    addrain_mask = shuffle_connected_components_torch(ldgp_img)
                    # addrain_mask = (ldgp_img)
                    net_output = addrain_model(rainy_images, addrain_mask)
                    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    plt.imsave(os.path.join(save_path,"test_addrain" + ".png"), denoised)

        if skip_batch:
            print("in another process, the training for this image is done.")
            continue
        with torch.no_grad():
            net_output = model(rainy_images)
            # net_output = pseudo_model(rainy_images)
            # print("[warning]: now use stage 2 (pseudo model) to do test!")
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