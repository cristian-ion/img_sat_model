import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from solution.image_utils.image_gray import binarize_grayscale


CLASSES = ["building"]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
INRIA_NAMECODE = "inria"
MAJOR_VERSION = 1
IMAGE_HEIGHT = 572
IMAGE_WIDTH = 572
ROOT_PATH = (
    "/Users/cristianion/Desktop/visual_recognition_train/inria/AerialImageDataset"
)
TRAIN_IMG_DIR = "/Users/cristianion/Desktop/visual_recognition_train/inria/AerialImageDataset/train/images"
TRAIN_MASK_DIR = "/Users/cristianion/Desktop/visual_recognition_train/inria/AerialImageDataset/train/gt"
VAL_IMG_DIR = "/Users/cristianion/Desktop/visual_recognition_train/inria/AerialImageDataset/val/images"
VAL_MASK_DIR = "/Users/cristianion/Desktop/visual_recognition_train/inria/AerialImageDataset/val/gt"
IMG_EXT = "tif"
MASK_EXT = "tif"

TRAIN_TRANSFORMS = A.Compose(
    [
        # A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

VAL_TRANSFORMS = A.Compose(
    [
        # A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


class InriaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

        self.images = [f for f in self.images if f.split(".")[-1] == IMG_EXT]
        self.masks = [f for f in self.masks if f.split(".")[-1] == MASK_EXT]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = None
        mask = None
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = binarize_grayscale(mask)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class InriaTrainValData:
    def __init__(self) -> None:
        self.train_transform = TRAIN_TRANSFORMS
        self.val_transform = VAL_TRANSFORMS

        self._trainset = InriaDataset(
            image_dir=TRAIN_IMG_DIR,
            mask_dir=TRAIN_MASK_DIR,
            transform=self.train_transform,
        )

        self._valset = InriaDataset(
            image_dir=VAL_IMG_DIR,
            mask_dir=VAL_MASK_DIR,
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
    def major_version(self):
        return MAJOR_VERSION
