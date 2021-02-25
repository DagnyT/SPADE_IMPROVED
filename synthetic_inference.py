import argparse
import yaml
import sys
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms
import torch
sys.path.append(".")
from utils import util
from dataset import make_data_loader

from models import build_model
from utils import settings
import numpy as np


def convert_labels(label_tensor):
    label_tensor[label_tensor == 0] = 1
    label_tensor[label_tensor == 255] = 2
    label_tensor[label_tensor == 125] = 3

    return label_tensor

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference for synthetic data - SPADE model')
    parser.add_argument('--path_ymlfile', type=str, default='configs/inference.yaml', help='Path to yaml file.')

    opt = parser.parse_args()

    with open(opt.path_ymlfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    _device = settings.initialize_cuda_and_logging(cfg)

    train_loader, val_loader = make_data_loader(cfg)

    model = build_model(cfg)

    input_folders = ['/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset/dislocations/segmentation_left_image',
                     '/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset/dislocations/segmentation_right_image']

    output_folders = ['/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset/dislocations/left_image/',
                      '/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset/dislocations/right_image/']

    style_directory_train = '/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/dislocations_segmentation_dataset/train_img/'
    style_directory_test = '/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/dislocations_segmentation_dataset/val_img/'

    style_images_train = os.listdir(style_directory_train)
    style_images_test = os.listdir(style_directory_test)

    for idx, folder_ in enumerate(input_folders):
        labels = os.listdir(folder_)
        for label_ in tqdm(labels):

            img_name = np.random.choice(style_images_train, 1)[0]
            label = Image.open(os.path.join(folder_, label_)).convert('L')

            image = Image.open(os.path.join(style_directory_train, img_name))
            image = image.convert('RGB')

            image = transforms.functional.resize(image, (cfg['TRAINING']['IMAGE_SIZE_W'], cfg['TRAINING']['IMAGE_SIZE_H']), Image.BICUBIC)
            label = transforms.functional.resize(label, (cfg['TRAINING']['IMAGE_SIZE_W'], cfg['TRAINING']['IMAGE_SIZE_H']), Image.NEAREST)

            label = convert_labels(np.array(label))

            image = transforms.functional.to_tensor(np.array(image))

            image_tensor = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

            label_tensor = torch.from_numpy(label).unsqueeze(0)

            data = {'label': label_tensor.unsqueeze(0),
                          'image': image_tensor.unsqueeze(0),
                          'path': os.path.join(folder_, label_) }

            fake_image = model.forward(data, 'inference')

            fake_image = util.tensor2im(fake_image, tile=False)

            Image.fromarray(fake_image[0]).save(output_folders[idx]+label_)
