import torch
import torch.nn as nn
import os
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.base_network import BaseNetwork
import numpy as np
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, cfg):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = cfg['TRAINING']['ENCODER']['NDF']
        norm_layer = get_nonspade_norm_layer(cfg, cfg['TRAINING']['NORM_E'])
        self.layer1 = norm_layer(nn.Conv2d(cfg['TRAINING']['OUTPUT_NC'], ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if cfg['TRAINING']['CROP_SIZE'] >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
            self.layer7 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.cfg = cfg
        self.print_network()

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.cfg['TRAINING']['CROP_SIZE'] >= 256:
            x = self.layer6(self.actvn(x))
            x = self.layer7(self.actvn(x))

        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

class VAE(nn.Module):
    def __init__(self, cfg, image_channels=3):
        super(VAE, self).__init__()
        ndf = cfg['TRAINING']['ENCODER']['NDF']
        kw = 3

        self.encoder = ConvEncoder(cfg)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, ndf * 8, kernel_size=kw, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ndf * 8, ndf * 8, kernel_size=kw, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ndf * 8, ndf * 8, kernel_size=kw, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=kw, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=kw, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ndf * 2, ndf * 1, kernel_size=kw, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ndf * 1, ndf * 1, kernel_size=kw, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ndf, 3, kernel_size=kw+1, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        return z

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.bottleneck(mu, logvar)

        decode = self.decoder(z)
        return decode, mu, logvar

    def save(self, epoch, net, cfg):
        label = 'vae'
        save_filename = '%s_net_%s.pth' % (epoch, label)
        save_path = os.path.join(cfg['LOGGING']['LOG_DIR'], cfg['TRAINING']['EXPERIMENT_NAME'], save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        if cfg['TRAINING']['GPU_ID'] == 0 and torch.cuda.is_available():
            net.cuda()

    def load(self, epoch, net, cfg):
        label = 'vae'
        save_filename = '%s_net_%s.pth' % (epoch, label)
        save_dir = os.path.join(cfg['LOGGING']['LOG_DIR'], cfg['TRAINING']['EXPERIMENT_NAME'])
        save_path = os.path.join(save_dir, save_filename)
        weights = torch.load(save_path)
        net.load_state_dict(weights)
        return net