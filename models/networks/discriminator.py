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

class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        for i in range(cfg['TRAINING']['NUM_D']):
            subnetD = self.create_single_discriminator(cfg)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, cfg):
        subarch = self.cfg['TRAINING']['NET_D_SUB_ARCH']
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(cfg)
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

        get_intermediate_features = not self.cfg['TRAINING']['NO_GAN_FEAT_LOSSop']
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]