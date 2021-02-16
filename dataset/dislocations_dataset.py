from PIL import Image

import numpy as np

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,RandomShadow
)

class DislocationsDataset():

    def get_paths(self, cfg):
        label_paths, image_paths = None, None
        return label_paths, image_paths

    def __init__(self, cfg, is_train = True):
        self.cfg = cfg
        self.is_train = is_train
        self.label_paths, self.image_paths = self.get_paths(cfg)
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
            label_tensor = Image.fromarray(label_tensor)

        label_tensor = self.convert_labels(np.array(label_tensor))
        image_tensor = Image.fromarray(image)/255.0

        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path }

        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size