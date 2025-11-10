import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = nn.Conv2d(in_ch, out_ch, 5, 2, 2, dilation=1, groups=1, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = nn.Conv2d(in_ch, out_ch, 7, 2, 3, dilation=1, groups=1, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, dilation=1, groups=1, bias=conv_bias)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, dilation=1, groups=1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        h = self.conv(input)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h


class UNet(nn.Module):
    def __init__(self, layer_size=4, input_channels=3, output_channels=3, upsampling_mode='nearest', is_target=False):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)
        self.is_target = is_target
        if is_target:
            self.alpha = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(self.layer_size)
            ])

            self.alpha_dec = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(self.layer_size)
            ])
    def forward(self, input, aux_model=None):
        h_dict = {}  # for the output of enc_N
        h_dict['h_0']= input
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            if aux_model is not None and i != 1:
                target_feature = getattr(self, l_key)(h_dict[h_key_prev])
                aux_feature = getattr(aux_model, l_key)(h_dict[h_key_prev])
                h_dict[h_key] = target_feature + self.alpha[i - 1] * aux_feature
            else :
                h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])

            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h = h_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            if aux_model is not None and i != 1:
                target_feature = getattr(self, dec_l_key)(h)
                aux_feature = getattr(aux_model, dec_l_key)(h)
                h = target_feature + self.alpha_dec[i - 1] * aux_feature
            else:
                h = getattr(self, dec_l_key)(h)

        return h

    def train(self, mode=True):
        
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()

class Seg_UNet(nn.Module):
    def __init__(self, layer_size=4, input_channels=3, output_channels=3, upsampling_mode='nearest', is_target=False):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, output_channels,
                              bn=False, activ=None, conv_bias=True)
        self.is_target = is_target
        if is_target:
            self.alpha = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(self.layer_size)
            ])

            self.alpha_dec = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(self.layer_size)
            ])
        self.sigmoid = nn.Sigmoid()
    def forward(self, input, aux_model=None):
        h_dict = {}  # for the output of enc_N
        h_dict['h_0']= input
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            if aux_model is not None and i != 1:
                target_feature = getattr(self, l_key)(h_dict[h_key_prev])
                aux_feature = getattr(aux_model, l_key)(h_dict[h_key_prev])
                h_dict[h_key] = target_feature + self.alpha[i - 1] * aux_feature
            else :
                h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])

            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h = h_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            if aux_model is not None and i != 1:
                target_feature = getattr(self, dec_l_key)(h)
                aux_feature = getattr(aux_model, dec_l_key)(h)
                h = target_feature + self.alpha_dec[i - 1] * aux_feature
            else:
                h = getattr(self, dec_l_key)(h)
        
        h = self.sigmoid(h)

        return h

    def train(self, mode=True):
        
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()