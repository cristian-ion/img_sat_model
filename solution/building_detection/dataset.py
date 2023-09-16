import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class MuBuildingsSegmentationDataset(Dataset):
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


if __name__ == "__main__":
    segm_dataset_train = MuBuildingsSegmentationDataset(
        image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train",
        mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train_labels",
        transform=None)

    segm_dataset_val = MuBuildingsSegmentationDataset(
        image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val",
        mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val_labels",
        transform=None)

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