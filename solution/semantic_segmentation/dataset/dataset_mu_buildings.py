import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

CLASSES = ["building"]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 8
MU_BUILDINGS_NAMECODE = "mub"


class MUBuildingsDataset(Dataset):
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


class MUBTrainValData:
    def __init__(self) -> None:
        IMAGE_HEIGHT = 512
        IMAGE_WIDTH = 512

        self.train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
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
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        self._trainset = MUBuildingsDataset(
            image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train",
            mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train_labels",
            transform=self.train_transform,
        )

        self._valset = MUBuildingsDataset(
            image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val",
            mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val_labels",
            transform=self.val_transform,
        )

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
    def namecode(self):
        return MU_BUILDINGS_NAMECODE


if __name__ == "__main__":
    """
    Example of usage.
    """
    segm_dataset_train = MUBuildingsDataset(
        image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train",
        mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train_labels",
        transform=None,
    )

    segm_dataset_val = MUBuildingsDataset(
        image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val",
        mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val_labels",
        transform=None,
    )

    train_loader = DataLoader(segm_dataset_train, batch_size=8, shuffle=True)
    val_loader = DataLoader(segm_dataset_val, batch_size=8, shuffle=False)

    it = iter(segm_dataset_train)
    for i in range(5):
        train_features, train_labels = next(it)
        print(train_features)
        print(f"Feature batch shape: {train_features.shape}")
        print(f"Labels batch shape: {train_labels.shape}")
        img = train_features
        plt.imshow(img)
        plt.show()
        img = train_labels
        plt.imshow(img)
        plt.show()
