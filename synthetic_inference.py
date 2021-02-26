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


def preprocess_input(cfg, image, label):

    image = transforms.functional.resize(image, (cfg['TRAINING']['IMAGE_SIZE_W'], cfg['TRAINING']['IMAGE_SIZE_H']),
                                         Image.BICUBIC)
    label = transforms.functional.resize(label,
                                              (cfg['TRAINING']['IMAGE_SIZE_W'], cfg['TRAINING']['IMAGE_SIZE_H']),
                                              Image.NEAREST)

    label = convert_labels(np.array(label))

    image = transforms.functional.to_tensor(np.array(image))

    image_tensor = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    label = torch.from_numpy(label).unsqueeze(0)

    return image_tensor, label


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

    input_folders_test = ['/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset_test/dislocations/segmentation_left_image',
                     '/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset_test/dislocations/segmentation_right_image']


    output_folders = ['/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset/dislocations/left_image/',
                      '/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset/dislocations/right_image/']

    output_folders_test = ['/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset_test/dislocations/left_image/',
                      '/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/Synthetic_Dislocations_V2/dislocations_dataset_test/dislocations/right_image/']

    for folder_ in output_folders_test:
        if not os.path.exists(folder_):
            os.makedirs(folder_)

    for folder_ in output_folders:
        if not os.path.exists(folder_):
            os.makedirs(folder_)


    style_directory_train = '/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/dislocations_segmentation_dataset/train_img/'
    style_directory_test = '/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/dislocations_segmentation_dataset/val_img/'

    style_images_train = os.listdir(style_directory_train)
    style_images_test = os.listdir(style_directory_test)


    # test

    labels = os.listdir(input_folders_test[0])
    for label_ in tqdm(labels):

        image = Image.open(os.path.join(style_directory_test, img_name))
        image = image.convert('RGB')

        img_name = np.random.choice(style_images_test, 1)[0]

        label_left = Image.open(os.path.join(input_folders_test[0], label_)).convert('L')
        label_right = Image.open(os.path.join(input_folders_test[1], label_.replace('LEFT','RIGHT'))).convert('L')

        image_tensor, label_tensor_left = preprocess_input(cfg, image, label_left)
        _, label_tensor_right = preprocess_input(cfg, image, label_right)

        seed = np.random.randint(1,100000)
        data = {'label': label_tensor_left.unsqueeze(0),
                          'image': image_tensor.unsqueeze(0),
                          'path': os.path.join(input_folders_test[0], label_),

                    'seed':seed}

        fake_image_left = model.forward(data, 'inference')

        fake_image_left = util.tensor2im(fake_image_left, tile=False)

        Image.fromarray(fake_image_left[0]).save(output_folders_test[0]+label_)

        data = {'label': label_tensor_right.unsqueeze(0),
                          'image': image_tensor.unsqueeze(0),
                          'path': os.path.join(input_folders_test[1], label_.replace('LEFT','RIGHT')),

                    'seed':seed }

        fake_image_right = model.forward(data, 'inference')
        fake_image_right = util.tensor2im(fake_image_right, tile=False)
        Image.fromarray(fake_image_right[0]).save(output_folders_test[1]+label_.replace('LEFT','RIGHT'))

    # train
    labels = os.listdir(input_folders[0])
    for label_ in tqdm(labels):

            img_name = np.random.choice(style_images_train, 1)[0]

            label_left = Image.open(os.path.join(input_folders[0], label_)).convert('L')
            label_right = Image.open(os.path.join(input_folders[1], label_.replace('LEFT','RIGHT'))).convert('L')

            image = Image.open(os.path.join(style_directory_train, img_name))
            image = image.convert('RGB')

            image_tensor, label_tensor_left = preprocess_input(cfg, image, label_left)
            _, label_tensor_right = preprocess_input(cfg, image, label_right)

            seed = np.random.randint(1,100000)
            data = {'label': label_tensor_left.unsqueeze(0),
                          'image': image_tensor.unsqueeze(0),
                          'path': os.path.join(input_folders[0], label_),

                    'seed':seed}

            fake_image_left = model.forward(data, 'inference')

            fake_image_left = util.tensor2im(fake_image_left, tile=False)

            Image.fromarray(fake_image_left[0]).save(output_folders[0]+label_)

            data = {'label': label_tensor_right.unsqueeze(0),
                          'image': image_tensor.unsqueeze(0),
                          'path': os.path.join(input_folders[1], label_.replace('LEFT','RIGHT')),

                    'seed':seed }

            fake_image_right = model.forward(data, 'inference')

            fake_image_right = util.tensor2im(fake_image_right, tile=False)

            Image.fromarray(fake_image_right[0]).save(output_folders[1]+label_.replace('LEFT','RIGHT'))



