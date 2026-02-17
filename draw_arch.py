import numpy as np 
import os
import torch
import matplotlib.pyplot as plt
from mask_aug import shuffle_connected_components_torch, shuffle_connected_components_torch_preserve_gray


def fixed_crop(x, top=40, left=10, h=256, w=256):
    return x[:, :, top:top+h, left:left+w]

def draw_stage_1(sdr_images_target, rainy_images, ldgp_img, addrain_model, save_path):
    arc_img_to_addrain = fixed_crop(sdr_images_target)
    arc_ori_img = fixed_crop(rainy_images)
    arc_addrain_mask = fixed_crop(ldgp_img)
    with torch.no_grad():
        arc_net_output = addrain_model(arc_img_to_addrain, arc_addrain_mask, arc_img_to_addrain)
    arc_addrain_result = np.clip(arc_net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"arc_addrain_reuslt" + ".png"), arc_addrain_result)
    arc_pseudo_label = np.clip(arc_img_to_addrain[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"arc_addrain_pseudo_label" + ".png"), arc_pseudo_label)
    arc_ori_img = np.clip(arc_ori_img[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"arc_addrain_origin_img" + ".png"), arc_ori_img)
    arc_addrain_mask = arc_addrain_mask[0, 0].detach().cpu().numpy()  # (H, W)
    arc_addrain_mask = np.clip(arc_addrain_mask, 0, 1)
    plt.imsave(os.path.join(save_path,"arc_addrain_mask" + ".png"), arc_addrain_mask, cmap="gray")


def draw_stage_2(ldgp_img, rainy_images, addrain_model, save_path, initial_model):
    arc_ori_img = fixed_crop(rainy_images)
    arc_addrain_mask = fixed_crop(ldgp_img, top=60, left=200)
    arc_addrain_mask_2 = shuffle_connected_components_torch_preserve_gray(arc_addrain_mask)

    with torch.no_grad():
        arc_net_output_1 = addrain_model(arc_ori_img, arc_addrain_mask, arc_ori_img)
        arc_net_output_2 = addrain_model(arc_ori_img, arc_addrain_mask_2, arc_ori_img)
        derain_output_1 = initial_model(arc_net_output_1)
        derain_output_2 = initial_model(arc_net_output_2)

    arc_net_output_1 = np.clip(arc_net_output_1[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"addrain_result_stage_2_output1" + ".png"), arc_net_output_1)
    arc_net_output_2 = np.clip(arc_net_output_2[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"addrain_result_stage_2_output2" + ".png"), arc_net_output_2)

    derain_output_1 = np.clip(derain_output_1[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"derain_result_stage_2_output1" + ".png"), derain_output_1)
    derain_output_2 = np.clip(derain_output_2[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"derain_result_stage_2_output2" + ".png"), derain_output_2)

    arc_addrain_mask = arc_addrain_mask[0, 0].detach().cpu().numpy()  # (H, W)
    arc_addrain_mask = np.clip(arc_addrain_mask, 0, 1)
    plt.imsave(os.path.join(save_path,"arc_stage_2_addrain_mask" + ".png"), arc_addrain_mask, cmap="gray")

    arc_addrain_mask_2 = arc_addrain_mask_2[0, 0].detach().cpu().numpy()  # (H, W)
    arc_addrain_mask_2 = np.clip(arc_addrain_mask_2, 0, 1)
    plt.imsave(os.path.join(save_path,"arc_stage_2_addrain_mask_2" + ".png"), arc_addrain_mask_2, cmap="gray")

    arc_ori_img_for_the_addrain_mask = fixed_crop(rainy_images, top=60, left=200)
    arc_ori_img_for_the_addrain_mask = np.clip(arc_ori_img_for_the_addrain_mask[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"arc_stage_2_ori_img_for_the_addrain_mask" + ".png"), arc_ori_img_for_the_addrain_mask)


def draw_stage_3(ldgp_img, rainy_images, addrain_model, save_path, initial_model, model):
    arc_ori_img = fixed_crop(rainy_images)
    rain_mask = fixed_crop(ldgp_img)
    arc_addrain_mask = fixed_crop(ldgp_img, top=60, left=200)
    arc_addrain_mask_2 = shuffle_connected_components_torch_preserve_gray(arc_addrain_mask)

    with torch.no_grad():
        pseudo_label_model_output = initial_model(arc_ori_img)
        pseudo_label = arc_ori_img * (1 - rain_mask) + pseudo_label_model_output * (rain_mask)
        arc_net_output_1 = addrain_model(arc_ori_img, arc_addrain_mask, arc_ori_img)
        arc_net_output_2 = addrain_model(arc_ori_img, arc_addrain_mask_2, arc_ori_img)
        derain_output_1 = model(arc_net_output_1)
        derain_output_2 = model(arc_net_output_2)
    
    pseudo_label = np.clip(pseudo_label[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"stage3_after_fusion" + ".png"), pseudo_label)

    derain_output_1 = np.clip(derain_output_1[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"stage3_derain_output_1" + ".png"), derain_output_1)

    derain_output_2 = np.clip(derain_output_2[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,"stage3_derain_output_2" + ".png"), derain_output_2)