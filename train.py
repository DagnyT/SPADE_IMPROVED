import os
import argparse
import yaml
import torch
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
    start_epoch, end_epoch = cfg['TRAINING']['START_EPOCH'], cfg['TRAINING']['END_EPOCHS']
    count = 0
    for epoch in tqdm(range(start_epoch, end_epoch + 1),total = end_epoch + 1):

        print('This is %d-th epoch' % epoch)

        for batch_idx, data in enumerate(train_loader):

            cur_iter = epoch * len(train_loader) + batch_idx

            image, label = model.preprocess_input(opt, data)

            model.module.netG.zero_grad()
            loss_G, generated = model(image, label, "generator")
            g_loss = sum(loss_G.values()).mean()
            g_loss.backward()
            optimizer_G.step()

            model.module.netD.zero_grad()
            loss_D = model(image, label, "discriminator")
            d_loss = sum(loss_D.values()).mean()
            d_loss.backward()
            optimizer_D.step()

            count+=1
            if cfg['LOGGING']['ENABLE_LOGGING']:
                tb_logger.add_scalars_to_tensorboard('Train', epoch, cur_iter, loss.item(), metrics.value())

            if cur_iter % cfg['LOGGING']['FID'] == 0 and cur_iter > 0:
                is_best = fid_computer.update(model, cur_iter)

            if cfg['VISUALIZER']['ENABLE'] == 1:

                visuals = OrderedDict([('input_label', data['label']),
                                       ('synthesized_image', generated),
                                       ('real_image', data['image'])])
                visualizer.display_current_results(visuals, epoch, cur_iter)


        if epoch % cfg['LOGGING']['LOG_INTERVAL'] == 0:

            total_test_loss = do_validate(epoch, model, val_loader, loss_func, _device, nms)
            logger.log_string('test loss for epoch {} : {}\n'.format(epoch, total_test_loss))
            logger.log_string('Mean repeatability for epoch {} : {}\n'.format(epoch, metrics.repeatability/metrics.count))

            print('epoch %d total test loss = %.3f' % (epoch, total_test_loss))

        if epoch % cfg['TRAINING']['SAVE_MODEL_STEP'] == 0:
            savefilename = os.path.join(cfg['TRAINING']['MODEL_DIR'], str(epoch) + 'glampoints.tar')

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(train_loader.dataset),
            }, savefilename)

            print('model is saved: {} - {}'.format(epoch, savefilename))

        model.update_learning_rate(epoch, optimizer_G, optimizer_D)

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training improved SPADE model')
    parser.add_argument('--path_ymlfile', type=str,default='configs/training.yml', help='Path to yaml file.')
    opt = parser.parse_args()

    with open(opt.path_ymlfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    _device = settings.initialize_cuda_and_logging(cfg)

    train_loader, val_loader = make_data_loader(cfg)

    model = build_model(cfg)
    model.module = model
    model.to(_device)

    optimizer_G, optimizer_D = model.create_optimizers(cfg)

    visualizer = visualizer.Visualizer(cfg)

    fid_computer = fid_scores.fid_pytorch(cfg, val_loader)

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