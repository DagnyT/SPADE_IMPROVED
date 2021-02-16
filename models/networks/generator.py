"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE
from models.networks.base_network import BaseNetwork
from linear_attention_transformer import ImageLinearAttention

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fn(x) * self.g

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out



attn = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, cfg):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in cfg['TRAINING']['NORM_G']:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        semantic_nc = cfg['TRAINING']['LABEL_NC'] + \
            (1 if cfg['TRAINING']['CONTAINS_DONT_CARE'] else 0) + \
            (0 if cfg['TRAINING']['NO_INSTANCE'] else 1)

        # define normalization layers

        spade_config_str = cfg['TRAINING']['NORM_G'].replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADEGenerator(BaseNetwork):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        nf = cfg['TRAINING']['NGF']

        self.sw, self.sh = self.compute_latent_vector_size(cfg)

        self.attn = attn(16 * nf)
        if cfg['USE_VAE']:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(self.cfg['TRAINING']['Z_DIM'], 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z

            semantic_nc = cfg['TRAINING']['LABEL_NC'] + \
                          (1 if cfg['TRAINING']['CONTAINS_DONT_CARE'] else 0) + \
                          (0 if cfg['TRAINING']['NO_INSTANCE'] else 1)

            self.fc = nn.Conv2d(semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, cfg)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, cfg)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, cfg)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, cfg)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, cfg)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, cfg)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, cfg)

        final_nc = nf

        if cfg['TRAINING']['NUM_UPSAMPLING_LAYERS'] == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, cfg)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        self.print_network()

    def compute_latent_vector_size(self, cfg):
        if cfg['TRAINING']['NUM_UPSAMPLING_LAYERS'] == 'normal':
            num_up_layers = 5
        elif cfg['TRAINING']['NUM_UPSAMPLING_LAYERS'] == 'more':
            num_up_layers = 6
        elif cfg['TRAINING']['NUM_UPSAMPLING_LAYERS'] == 'most':
            num_up_layers = 7
        else:
            raise ValueError('num_upsampling_layers [%s] not recognized' %
                             cfg['TRAINING']['NUM_UPSAMPLING_LAYERS'])

        sw = cfg['TRAINING']['CROP_SIZE'] // (2**num_up_layers)
        sh = round(sw / cfg['TRAINING']['ASPECT_RATIO'])

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.cfg['USE_VAE']:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.cfg['TRAINING']['Z_DIM'],
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.cfg['TRAINING']['NGF'], self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.attn(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.cfg['TRAINING']['NUM_UPSAMPLING_LAYERS'] == 'more' or \
           self.cfg['TRAINING']['NUM_UPSAMPLING_LAYERS'] == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.cfg['TRAINING']['NUM_UPSAMPLING_LAYERS'] == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x