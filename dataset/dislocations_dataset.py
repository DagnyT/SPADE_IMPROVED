from PIL import Image
from torchvision import transforms

import numpy as np
import os
import torch

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,RandomShadow
)

class DislocationsDataset():

    def get_paths(self, cfg, mode):

        if mode == 'train':
            path = cfg['INPUT']['TRAIN']
        elif mode == 'val':
            path = cfg['INPUT']['VAL']
        else:
            path = cfg['INPUT']['TEST']

        path_img = os.path.join(path, mode+"_img")

        images = sorted(os.listdir(path_img))
        return images, path_img

    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.mode = mode

        self.images, self.path_img = self.get_paths(cfg, mode)
        self.dataset_size = len(self.images)

        print('Dataset size: {}'.format(self.dataset_size))

    def convert_labels(self, label_tensor):

        label_tensor[label_tensor == 0] = 1
        label_tensor[label_tensor == 255] = 2
        label_tensor[label_tensor == 125] = 3

        return label_tensor

    def deconvert_labels(self, label_tensor):

        label_tensor[label_tensor == 1] = 0
        label_tensor[label_tensor == 2] = 255
        label_tensor[label_tensor == 3] = 125

        return label_tensor


    def strong_aug(self, p=0.5):
        return Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.01),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.5),
            OneOf([
                RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
            ], p=0.5),
        ], p=p)


    def __getitem__(self, index):

        label_path = self.path_img.replace(self.mode+'_img',self.mode+"_label")
        img_name = self.images[index]

        label = Image.open(os.path.join(label_path, img_name)).convert('L')
        image = Image.open(os.path.join(self.path_img, img_name))
        image = image.convert('RGB')

        image = transforms.functional.resize(image, (self.cfg['TRAINING']['IMAGE_SIZE_W'], self.cfg['TRAINING']['IMAGE_SIZE_H']), Image.BICUBIC)
        label = transforms.functional.resize(label, (self.cfg['TRAINING']['IMAGE_SIZE_W'], self.cfg['TRAINING']['IMAGE_SIZE_H']), Image.NEAREST)

        if self.mode == 'train':

            augmentation = self.strong_aug(p=0.4)
            data_l = {"image": np.array(image), "mask": np.array(label).squeeze()}

            augmented_l = augmentation(**data_l)

            image, label_tensor = augmented_l["image"], augmented_l["mask"]
            label = Image.fromarray(label_tensor)

        label = self.convert_labels(np.array(label))

        image = transforms.functional.to_tensor(np.array(image))
        # normalize
        image_tensor = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        label_tensor = torch.from_numpy(label).unsqueeze(0)

        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': os.path.join(self.path_img, img_name) }

        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size