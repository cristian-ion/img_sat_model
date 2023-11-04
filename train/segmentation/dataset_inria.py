import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
import cv2

from train.image_utils.image_gray import binarize_grayscale, crop
from constants import REPO_DIR

CLASSES = ["building"]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
INRIA_NAMECODE = "inria"
MAJOR_VERSION = 1
ROOT_PATH = "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset"

# TRAIN_IMG_DIR = (
#     "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/train/images"
# )
# TRAIN_MASK_DIR = (
#     "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/train/gt"
# )
# VAL_IMG_DIR = (
#     "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/val/images"
# )
# VAL_MASK_DIR = (
#     "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/val/gt"
# )

CROP_SIZE = 572
MASK_SIZE = 388
TRAIN_IMG_DIR = f"{REPO_DIR}/_inria_train_images/{CROP_SIZE}/train/img"
TRAIN_MASK_DIR = f"{REPO_DIR}/_inria_train_images/{CROP_SIZE}/train/gt"
VAL_IMG_DIR = f"{REPO_DIR}/_inria_train_images/{CROP_SIZE}/val/img"
VAL_MASK_DIR = f"{REPO_DIR}/_inria_train_images/{CROP_SIZE}/val/gt"

IMG_EXT = "png"
MASK_EXT = "png"


#
# https://albumentations.ai/docs/api_reference/augmentations/dropout/mask_dropout/
# https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
#
TRAIN_TRANSFORMS = A.Compose(
    transforms=[
        # A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.01, rotate_limit=5, p=0.5),
        # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.1),
        # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Transpose(p=0.2),
        A.RandomRotate90(p=0.2),
        A.VerticalFlip(p=0.2),
        A.HorizontalFlip(p=0.2),
        A.Normalize(
            # mean=(0, 0, 0),
            mean=(0.485, 0.456, 0.406),
            # std=(1, 1, 1),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            always_apply=True,
        ),
        ToTensorV2(),
    ],
    is_check_shapes=False,
)

VAL_TRANSFORMS = A.Compose(
    transforms=[
        A.Normalize(
            mean=(0, 0, 0),
            std=(1, 1, 1),
            max_pixel_value=255.0,
            always_apply=True,
        ),
        ToTensorV2(),
    ],
    is_check_shapes=False,
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

        # image = np.array(Image.open(image_path).convert("RGB"))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = binarize_grayscale(mask)
        mask = crop(mask, border=92) # 92 = (572-388)//2

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class InriaTrainConfig:
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
        self.num_epochs = 100

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
