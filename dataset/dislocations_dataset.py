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
        path_lab = os.path.join(path, mode+"_label")

        images , labels = sorted(os.listdir(path_img)), sorted(os.listdir(path_lab))

        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))

        return images, labels

    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.is_train = False
        if mode == 'train':
            self.is_train = True

        self.label_paths, self.image_paths = self.get_paths(cfg, mode)
        size = len(self.label_paths)
        self.dataset_size = size

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
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.5),
            OneOf([
                RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
            ], p=0.5),
        ], p=p)


    def __getitem__(self, index):

        label_path = self.label_paths[index]
        image_path = self.image_paths[index]

        label = Image.open(label_path).convert('L')
        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.is_train:

            augmentation = self.strong_aug(p=0.4)
            data_l = {"image": np.array(image), "mask": np.array(label).squeeze()}

            augmented_l = augmentation(**data_l)

            image, label_tensor = augmented_l["image"], augmented_l["mask"]
            label = Image.fromarray(label_tensor)

        label = self.convert_labels(np.array(label))

        image_tensor = torch.from_numpy(np.array(image))
        label_tensor = torch.from_numpy(label).unsqueeze(0)

        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path }

        self.postprocess(input_dict)
        print(input_dict)
        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size