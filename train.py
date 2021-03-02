import argparse
import yaml
import sys
import time
from tqdm import tqdm
from collections import OrderedDict

sys.path.append(".")

from dataset import make_data_loader
from logger import build_logger

from models import build_model
from utils import settings, visualizer, fid_scores

def do_validate():
    return

def do_train(cfg, model, train_loader, val_loader, optimizer_G, optimizer_D, fid_computer, logger, tb_logger, _device):

    start_full_time = time.time()

    if cfg['LOGGING']['ENABLE_LOGGING']:
        logger.log_string(cfg)
    start_epoch, end_epoch = cfg['TRAINING']['START_EPOCH'], cfg['TRAINING']['N_ITER']+cfg['TRAINING']['N_ITER_DECAY']
    for epoch in tqdm(range(start_epoch, end_epoch + 1),total = end_epoch + 1):

        print('This is %d-th epoch' % epoch)

        for batch_idx, data in enumerate(train_loader):
            current_time = time.time()

            cur_iter = epoch * len(train_loader) + batch_idx

            optimizer_G.zero_grad()
            loss_G, generated = model(data, "generator")
            g_loss = sum(loss_G.values()).mean()
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            loss_D = model(data, "discriminator")
            d_loss = sum(loss_D.values()).mean()
            d_loss.backward()
            optimizer_D.step()

            if cfg['LOGGING']['ENABLE_LOGGING'] and cur_iter % cfg['LOGGING']['LOG_INTERVAL'] == 0:
                print('Train', epoch, cur_iter, g_loss, d_loss)
                tb_logger.add_scalars_to_tensorboard('Train', epoch, cur_iter, loss_G, loss_D)
            #
            if cur_iter % cfg['LOGGING']['FID'] == 0:
                is_best = fid_computer.update(model, cur_iter)

            if cfg['VISUALIZER']['ENABLE'] and cur_iter % cfg['VISUALIZER']['LOG_INTERVAL'] == 0:

                visuals = OrderedDict([('input_label', data['label']),
                                       ('synthesized_image', generated),
                                       ('real_image', data['image'])])
                visualizer.display_current_results(visuals, epoch, cur_iter)

                visualizer.print_current_errors(epoch, cur_iter,
                                                {**loss_G, **loss_D}, time.time() - current_time)

        if epoch % cfg['LOGGING']['SAVE_EVERY'] == 0 :
            model.save(epoch)
            print('model is saved: {} '.format(epoch))

        model.update_learning_rate(optimizer_G, optimizer_D, epoch)

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training improved SPADE model')
    parser.add_argument('--path_ymlfile', type=str, default='configs/training_effnet.yaml', help='Path to yaml file.')
    opt = parser.parse_args()

    with open(opt.path_ymlfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    _device = settings.initialize_cuda_and_logging(cfg)

    train_loader, val_loader = make_data_loader(cfg)

    model = build_model(cfg)

    optimizer_G, optimizer_D = model.create_optimizers(cfg)
    visualizer = visualizer.Visualizer(cfg)
    print('Fid initialization')
    fid_computer = None
    fid_computer = fid_scores.fid_pytorch(cfg, val_loader)

    print('Fid was initialized')

    logger, tb_logger = build_logger(cfg)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer_G, optimizer_D,
        fid_computer,
        logger,
        tb_logger,
        _device)

    print('Training was successfully finished.')