import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["building"]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
INRIA_NAMECODE = "inria"
VERSION = 1


class InriaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = None
        mask = None
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class InriaTrainValData:
    def __init__(self) -> None:
        IMAGE_HEIGHT = 256
        IMAGE_WIDTH = 256

        self.train_transform = A.Compose(
            [
                A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        self.val_transform = A.Compose(
            [
                A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        self._trainset = InriaDataset(
            image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train",
            mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train_labels",
            transform=self.train_transform,
        )

        self._valset = InriaDataset(
            image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val",
            mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val_labels",
            transform=self.val_transform,
        )

        self._criterion = torch.nn.BCEWithLogitsLoss()

    @property
    def trainset(self):
        return self._trainset

    @property
    def valset(self):
        return self._valset

    @property
    def num_classes(self):
        return len(CLASSES)

    @property
    def batch_size(self):
        return BATCH_SIZE

    @property
    def val_batch_size(self):
        return BATCH_SIZE

    @property
    def namecode(self):
        return INRIA_NAMECODE

    @property
    def criterion(self):
        return self._criterion

    @property
    def version(self):
        return VERSION
