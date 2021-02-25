"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import os
from models.networks.encoder import ConvEncoder
from models.networks.generator import SPADEGenerator
from models.networks.discriminator import MultiscaleDiscriminator
from models.networks.loss import GANLoss, VGGLoss, KLDLoss, GradLoss

class SpadeModel(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(cfg)

        self.old_lr = cfg['TRAINING']["LR"]

        if not cfg['IS_TRAINING']:
            self.inference_mode()

        if cfg['IS_TRAINING']:
            self.criterionGAN = GANLoss(
                cfg['TRAINING']['GAN_MODE'], tensor=self.FloatTensor, cfg=self.cfg)
            self.criterionFeat = torch.nn.L1Loss()
            if not cfg['TRAINING']['NO_VGG_LOSS']:
                self.criterionVGG = VGGLoss(cfg['TRAINING']['GPU_ID'])
            if cfg['USE_VAE']:
                self.KLDLoss = KLDLoss()
            self.GradLoss = GradLoss()

    def inference_mode(self):

        self.netG.eval()
        self.netE.eval()


    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _,_ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, cfg):
        G_params = list(self.netG.parameters())
        if cfg['USE_VAE']:
            G_params += list(self.netE.parameters())
        if cfg['IS_TRAINING']:
            D_params = list(self.netD.parameters())

        beta1, beta2 = cfg['TRAINING']['BETA1'], cfg['TRAINING']['BETA2']
        if cfg['TRAINING']['NO_TTUR']:
            G_lr, D_lr = cfg['TRAINING']['LR'], cfg['TRAINING']['LR']
        else:
            G_lr, D_lr = cfg['TRAINING']['LR']/ 2, cfg['TRAINING']['LR'] * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save_network(self, net, label, epoch, cfg):
        save_filename = '%s_net_%s.pth' % (epoch, label)
        save_path = os.path.join(cfg['LOGGING']['LOG_DIR'], cfg['TRAINING']['EXPERIMENT_NAME'], save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        if cfg['TRAINING']['GPU_ID'] == 0 and torch.cuda.is_available():
            net.cuda()

    def load_network(self, net, label, epoch, cfg):
        save_filename = '%s_net_%s.pth' % (epoch, label)
        save_dir = os.path.join(cfg['LOGGING']['LOG_DIR'], cfg['TRAINING']['EXPERIMENT_NAME'])
        save_path = os.path.join(save_dir, save_filename)
        weights = torch.load(save_path)
        net.load_state_dict(weights)
        return net

    def save(self, epoch):
        self.save_network(self.netG, 'G', epoch, self.cfg)
        self.save_network(self.netD, 'D', epoch, self.cfg)
        if self.cfg['USE_VAE']:
            self.save_network(self.netE, 'E', epoch, self.cfg)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, cfg):

        netG = SPADEGenerator(cfg)
        netD = MultiscaleDiscriminator(cfg) if cfg['IS_TRAINING'] else None
        netE = ConvEncoder(cfg) if cfg['USE_VAE'] else None

        netG.cuda()
        netE.cuda()

        if cfg['IS_TRAINING']:

            netD.cuda()
            netD.init_weights(cfg)

        netG.init_weights(cfg)
        netE.init_weights(cfg)

        if not cfg['IS_TRAINING'] or cfg['CONTINUE_TRAINING']:

            print('Loading networks from file')
            netG = self.load_network(netG, 'G', cfg['TRAINING']['WHICH_EPOCH'], cfg)
            if cfg['IS_TRAINING']:
                netD = self.load_network(netD, 'D', cfg['TRAINING']['WHICH_EPOCH'], cfg)
            if cfg['USE_VAE']:
                netE = self.load_network(netE, 'E', cfg['TRAINING']['WHICH_EPOCH'], cfg)
        print('Networks were initialized ')

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.cfg['TRAINING']['LABEL_NC'] + 1 if self.cfg['TRAINING']['CONTAINS_DONT_CARE'] \
            else self.cfg['TRAINING']['LABEL_NC']
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map.long(), 1.0)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss, Encoder_Loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.cfg['USE_VAE'])

        # G_losses['GradLoss'] = self.GradLoss(fake_image, real_image)*10

        if self.cfg['USE_VAE']:
            G_losses['KLD'] = KLD_loss
            # G_losses['Encoder_Loss'] = Encoder_Loss*10

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, input_semantics, True,
                                            for_discriminator=False)

        if not self.cfg['TRAINING']['NO_GAN_FEAT_LOSS']:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.cfg['TRAINING']['LAMBDA_FEAT'] / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.cfg['TRAINING']['NO_VGG_LOSS']:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.cfg['TRAINING']['LAMBDA_VGG']

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _,_ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, input_semantics, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, input_semantics, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.cfg['USE_VAE']:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.cfg['TRAINING']['LAMBDA_KLD']

        fake_image = self.netG(input_semantics, z=z)
        z_fake, mu_fake, logvar_fake = self.encode_z(fake_image)

        Encoder_Loss = (mu-mu_fake)**2 + (torch.exp(0.5 * logvar) - torch.exp(0.5 * logvar_fake))**2

        assert (not compute_kld_loss) or self.cfg['USE_VAE'], \
            "You cannot compute KLD loss if self.cfg['USE_VAE'] == False"

        return fake_image, KLD_loss, Encoder_Loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):

        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):

        if self.cfg['TRAINING']['NET_D_SUB_ARCH'] == 'oasis':
            if type(pred) == list:
                fake = []
                real = []
                for p in pred:
                    fake.append([p[:p.size(0) // 2]])
                    real.append([p[p.size(0) // 2:]])
                return fake, real

        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return True

    def update_learning_rate(self, optimizer_G, optimizer_D, epoch):
        if epoch > self.cfg['TRAINING']['N_ITER']:
            lrd = self.cfg['TRAINING']['LR'] / self.cfg['TRAINING']['N_ITER_DECAY']
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.cfg['TRAINING']['NO_TTUR']:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr