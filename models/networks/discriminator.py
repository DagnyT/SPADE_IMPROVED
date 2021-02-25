"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.base_network import BaseNetwork
import torch.nn.utils.spectral_norm as spectral_norm

import torch
import torch.nn as nn


class OASIS_Discriminator(BaseNetwork):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        sp_norm = spectral_norm
        output_channel = cfg['TRAINING']['OUTPUT_NC'] + 1
        self.channels = [7, 128, 128, 256, 512]
        self.body_up   = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(cfg['TRAINING']['N_LAYERS_D']):
            self.body_down.append(residual_block_D(self.channels[i], self.channels[i+1], cfg, -1, first=(i==0)))
        # decoder part
        self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2], cfg, 1))
        for i in range(1, cfg['TRAINING']['N_LAYERS_D']-1):
            self.body_up.append(residual_block_D(2*self.channels[-1-i], self.channels[-2-i], cfg, 1))
        self.body_up.append(residual_block_D(2*self.channels[1], 64, cfg, 1))
        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

    def forward(self, input):
        x = input
        #encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
        #decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](torch.cat((encoder_res[-i-1], x), dim=1))
        ans = self.layer_up_last(x)
        return ans


class residual_block_D(nn.Module):
    def __init__(self, fin, fout, cfg, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        norm_layer = spectral_norm
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        self.conv2 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s

class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        for i in range(cfg['TRAINING']['NUM_D']):
            subnetD = self.create_single_discriminator(cfg)
            self.add_module('discriminator_%d' % i, subnetD)
        self.print_network()

    def create_single_discriminator(self, cfg):
        subarch = self.cfg['TRAINING']['NET_D_SUB_ARCH']
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(cfg)

        elif subarch == 'oasis':
            netD = OASIS_Discriminator(cfg)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size num_D x n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.cfg['TRAINING']['NO_GAN_FEAT_LOSS']
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = cfg['TRAINING']['ENCODER']['NDF']
        input_nc = self.compute_D_input_nc(cfg)

        norm_layer = get_nonspade_norm_layer(cfg, cfg['TRAINING']['NORM_D'])
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, cfg['TRAINING']['N_LAYERS_D']):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == cfg['TRAINING']['N_LAYERS_D'] - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, cfg):
        input_nc = cfg['TRAINING']['LABEL_NC'] + cfg['TRAINING']['OUTPUT_NC']
        if cfg['TRAINING']['CONTAINS_DONT_CARE']:
            input_nc += 1
        if not cfg['TRAINING']['NO_INSTANCE']:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.cfg['TRAINING']['NO_GAN_FEAT_LOSS']
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]