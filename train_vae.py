from models.networks import vae
from dataset import make_data_loader
import torch
from tqdm import tqdm

import argparse
import yaml

def do_train(cfg, model, train_loader, val_loader, optimizer):

    start_epoch, end_epoch = cfg['TRAINING']['START_EPOCH'], cfg['TRAINING']['N_ITER']+cfg['TRAINING']['N_ITER_DECAY']

    for epoch in tqdm(range(start_epoch, end_epoch + 1),total = end_epoch + 1):

        for idx, data in enumerate(train_loader):
            data['image'] = data['image'].cuda()
            recon_images, mu, logvar = model(data['image'])
            loss, bce, kld = loss_fn(recon_images, data['image'], mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1,
                                                                        end_epoch + 1, loss.item() / cfg['TRAINING']['BATCH_SIZE'], bce.item() / cfg['TRAINING']['BATCH_SIZE'],
                                                                        kld.item() / cfg['TRAINING']['BATCH_SIZE'])
            print(to_print)

            if epoch % cfg['LOGGING']['SAVE_EVERY'] == 0:
                model.save(epoch, model, cfg)
                print('model is saved: {} '.format(epoch))


def loss_fn(recon_x, x, mu, logvar):
        MSE_loss = torch.nn.MSELoss()
        MSE = MSE_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD, MSE, KLD


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training improved SPADE model')
    parser.add_argument('--path_ymlfile', type=str, default='configs/training.yaml', help='Path to yaml file.')
    opt = parser.parse_args()

    with open(opt.path_ymlfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    model = vae.VAE(cfg)
    model.cuda()
    train_loader, val_loader = make_data_loader(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    do_train(cfg, model, train_loader, val_loader, optimizer)