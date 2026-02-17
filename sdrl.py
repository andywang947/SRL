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
from mask_aug import shuffle_connected_components_torch, shuffle_connected_components_torch_preserve_gray
import yaml

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
if not cfg["loss"]["stage_2"]["LRR_supervision_single"]:
    print("[warning]: for L_RR, use dual output to supervision, it's ablation for stage 2")
if not cfg["loss"]["stage_2"]["use_consistency"]:
    print("[warning]: didn't use consistency loss in stage 2, it's ablation")

if not cfg["loss"]["stage_3"]["L_REF_do_mask_fusion"]:
    print("[warning]: didn't use pseudo label mask fusion in stage 3, it's ablation")
if not cfg["loss"]["stage_3"]["LREF_supervision_single"]:
    print("[warning]: didn't use single supervision in stage 3, it's ablation")
if not cfg["loss"]["stage_3"]["use_consistency"]:
    print("[warning]: didn't use consistency loss in stage 3, it's ablation")

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

        initial_model = UNet()
        initial_model = initial_model.to(device)
        
        from addrain_network import AddRainNet, AddRainNet_test
        if opt.result_name == "test" and False:
            print("warning: now is the test mode")
            addrain_model = AddRainNet_test(input_channels=7, output_channels=3)        
        else:
            addrain_model = AddRainNet(input_channels=4, output_channels=3)
        addrain_model = addrain_model.to(device)

        addrain_optimizer = Adam(addrain_model.parameters(), lr=0.001)
        addrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(addrain_optimizer, T_max=epochs)
        addrain_model.train()


        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


        pseudo_optimizer = Adam(initial_model.parameters(), lr=0.001)
        pseudo_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pseudo_optimizer, T_max=epochs)

        inner_batch_size = 1
        rainy_images = rainy_images.to(device)
        sdr_images_target = sdr_images_target.to(device)
        ldgp_img = ldgp_img.to(device)

        model.train()
        Addrain_loader = Addrain_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=inner_batch_size)
        for epoch in tqdm(range(epochs)):
            addrain_model.train()
            for k, inner_batch in enumerate(islice(Addrain_loader, 50)):
                sdr_images, input_img, rain_mask, another_rain_mask, another_input_img = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)
                another_rain_mask = another_rain_mask.to(device)
                another_input_img = another_input_img.to(device)

                addrain_optimizer.zero_grad()

                zero_mask = torch.zeros_like(rain_mask)

                addrain_output_1 = addrain_model(sdr_images, rain_mask, input_img)                
                # addrain_output_2 = addrain_model(sdr_images, zero_mask, another_input_img)                
                loss_1 = torch.abs(addrain_output_1 - input_img).mean()
                # loss_2 = torch.abs(addrain_output_2 - sdr_images).mean()

                # loss = loss_1 + loss_2
                loss = loss_1

                loss.backward()
                addrain_optimizer.step()

            addrain_scheduler.step()
                    
            # inference
            addrain_model.eval()
            with torch.no_grad():
                if cfg['architecture_figure']:
                    from draw_arch import draw_stage_1
                    draw_stage_1(sdr_images_target, rainy_images, ldgp_img, addrain_model, save_path)

                if opt.result_name == "test":
                    img_to_addrain = torch.flip(rainy_images, dims=[2, 3]) 
                    addrain_mask = torch.flip(ldgp_img, dims=[2, 3])
                    # addrain_mask = shuffle_connected_components_torch(ldgp_img)

                    net_output = addrain_model(rainy_images, addrain_mask, img_to_addrain)
                    net_output = net_output[:,:,:h,:w]
                    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    plt.imsave(os.path.join(save_path,name[0]), denoised)

        skip_batch = False
        SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=inner_batch_size)
        for j in tqdm(range(epochs)):
            if os.path.exists(img_save_path) == True and opt.result_name != "test":
                skip_batch = True
                break
            initial_model.train()

            for k, inner_batch in enumerate(islice(SDR_loader, 50)):
                sdr_images, input_img, rain_mask, another_mask_1, another_input_img_1, another_mask_2, another_input_img_2 = inner_batch
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)
                another_mask_1 = another_mask_1.to(device)
                another_input_img_1 = another_input_img_1.to(device)
                another_mask_2 = another_mask_2.to(device)
                another_input_img_2 = another_input_img_2.to(device)

                with torch.no_grad():
                    addrain_input = addrain_model(input_img, another_mask_1, another_input_img_1)
                    another_mask_2 = shuffle_connected_components_torch(another_mask_1)
                    addrain_input_2 = addrain_model(input_img, another_mask_2, another_input_img_2)

                pseudo_net_output_1 = initial_model(addrain_input)
                pseudo_net_output_2 = initial_model(addrain_input_2)

                if cfg["loss"]["stage_2"]["Region_of_L_RR_use_non_rain"]:
                    reconstruction_loss_1 = ((torch.abs(pseudo_net_output_1 - input_img)) * (1 - rain_mask)).mean()
                    reconstruction_loss_2 = ((torch.abs(pseudo_net_output_2 - input_img)) * (1 - rain_mask)).mean()
                else:
                    reconstruction_loss_1 = (torch.abs(pseudo_net_output_1 - input_img)).mean()
                    reconstruction_loss_2 = (torch.abs(pseudo_net_output_2 - input_img)).mean()

                if cfg["loss"]["stage_2"]["LRR_supervision_single"]:
                    reconstruction_loss = reconstruction_loss_1
                else:
                    reconstruction_loss = 0.5 * (reconstruction_loss_1 + reconstruction_loss_2)
                # reconstruction_loss = reconstruction_loss_1
                if cfg["loss"]["stage_2"]["use_consistency"]:
                    consistency_loss = torch.abs(pseudo_net_output_1 - pseudo_net_output_2).mean()
                else:
                    consistency_loss = 0

                loss = reconstruction_loss + consistency_loss
                pseudo_optimizer.zero_grad()
                loss.backward()
                pseudo_optimizer.step()

            pseudo_scheduler.step()

            # inference
            initial_model.eval()
            if cfg['architecture_figure']:
                from draw_arch import draw_stage_2
                draw_stage_2(ldgp_img, rainy_images, addrain_model, save_path, initial_model)
            with torch.no_grad():
                if opt.result_name == "test":
                    net_output = initial_model(rainy_images)
                    net_output = net_output[:,:,:h,:w]
                    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    plt.imsave(os.path.join(save_path,name[0]), denoised)

        skip_batch = False
        for j in tqdm(range(epochs)):
            if os.path.exists(img_save_path) == True and opt.result_name != "test":
                skip_batch = True
                break
            model.train()
            initial_model.train()
            addrain_model.eval()

            for k, inner_batch in enumerate(islice(SDR_loader, 50)):
                sdr_images, input_img, rain_mask, another_mask_1, another_input_img_1, another_mask_2, another_input_img_2 = inner_batch
                
                sdr_images = sdr_images.to(device)
                input_img = input_img.to(device)
                rain_mask = rain_mask.to(device)
                another_mask_1 = another_mask_1.to(device)
                another_input_img_1 = another_input_img_1.to(device)
                another_mask_2 = another_mask_2.to(device)
                another_input_img_2 = another_input_img_2.to(device)

                with torch.no_grad():
                    addrain_input = addrain_model(input_img, another_mask_1, another_input_img_1)
                    another_mask_2 = shuffle_connected_components_torch(another_mask_1)
                    addrain_input_2 = addrain_model(input_img, another_mask_2, another_input_img_2)

                with torch.no_grad():
                    if cfg['teacher_weakly_aug']:
                        pseudo_net_output = initial_model(input_img)
                    else:
                        pseudo_net_output = initial_model(addrain_input)
                
                if cfg['student_strongly_aug']:
                    net_output_1 = model(addrain_input)
                else:
                    net_output_1 = model(input_img)
                net_output_2 = model(addrain_input_2)
    
                if cfg["loss"]["stage_3"]["L_REF_do_mask_fusion"]:
                    pseudo_label = input_img * (1 - rain_mask) + pseudo_net_output * (rain_mask)
                else:
                    pseudo_label = pseudo_net_output

                if cfg["loss"]["stage_3"]["LREF_supervision_single"]:
                    reconstruction_loss_1 = torch.abs(net_output_1 - pseudo_label).mean()
                    reconstruction_loss = reconstruction_loss_1 
                else:
                    reconstruction_loss_1 = torch.abs(net_output_1 - pseudo_label).mean()
                    reconstruction_loss_2 = torch.abs(net_output_2 - pseudo_label).mean()
                    reconstruction_loss = 0.5 * (reconstruction_loss_1 + reconstruction_loss_2)

                if cfg["loss"]["stage_3"]["use_consistency"]:
                    consistency_loss = (torch.abs(net_output_1 - net_output_2) * (1 - rain_mask)).mean()
                else:
                    consistency_loss = 0

                loss = reconstruction_loss + consistency_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # addrain_scheduler.step()
            # pseudo_scheduler.step()
            scheduler.step()
        
            # inference
            model.eval()

            if cfg['architecture_figure']:
                from draw_arch import draw_stage_3
                draw_stage_3(ldgp_img, rainy_images, addrain_model, save_path, initial_model, model)
            with torch.no_grad():
                if opt.result_name == "test":
                    # mask_feature = mask_model(rainy_images, return_bottleneck=True)
                    if cfg['use_stage3']:
                        net_output = model(rainy_images)
                    else:
                        net_output = initial_model(rainy_images)
                    # net_output = model(rainy_images)
                    net_output = net_output[:,:,:h,:w]
                    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    plt.imsave(os.path.join(save_path,name[0]), denoised)
                    # plt.imsave(os.path.join(save_path,"test_" + str(j) + ".png"), denoised)
                        # plt.imsave(os.path.join(save_path,"test_" + str(i) + ".png"), denoised)
                    addrain_mask = torch.flip(ldgp_img, dims=[2, 3])
                    addrain_mask = shuffle_connected_components_torch(ldgp_img)
                    # addrain_mask = (ldgp_img)
                    # net_output = addrain_model(rainy_images, addrain_mask)
                    # denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
                    # plt.imsave(os.path.join(save_path,"test_addrain" + ".png"), denoised)

        if skip_batch:
            print("in another process, the training for this image is done.")
            continue
        with torch.no_grad():
            if cfg['use_stage3']:
                net_output = model(rainy_images)
            else:
                print("[warning]: use the stage 2 derain model to output the result, it's a ablaiton")
                net_output = initial_model(rainy_images)
            # net_output = initial_model(rainy_images)
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