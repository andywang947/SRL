import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class AddRainNet(nn.Module):
    def __init__(self, layer_size=4, input_channels=3, output_channels=3, upsampling_mode='nearest'):
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
        self.dec_1 = PCBActiv(64 + input_channels, output_channels, bn=False, activ=None, conv_bias=True)
    def forward(self, x, mask):
        h_dict = {}  # for the output of enc_N
        # print("time to concate")
        model_input = torch.cat([x, mask],dim=1)
        h_dict['h_0']= model_input
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])  

            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h = h_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h = getattr(self, dec_l_key)(h)

        return h

    def train(self, mode=True):
        
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()        


class AddRainNet_test(nn.Module):
    def __init__(self, layer_size=4, input_channels=3, output_channels=3, upsampling_mode='nearest'):
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
        self.dec_1 = PCBActiv(64 + input_channels, output_channels, bn=False, activ=None, conv_bias=True)
    def forward(self, x, mask):
        h_dict = {}  # for the output of enc_N
        # print("time to concate")
        model_input = torch.cat([x, mask],dim=1)
        h_dict['h_0']= model_input
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])  

            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h = h_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h = getattr(self, dec_l_key)(h)

        return h

    def train(self, mode=True):
        
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()     