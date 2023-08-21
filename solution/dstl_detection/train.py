import csv
import json
import os
import sys

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
import tifffile as tiff
import torch
import torch.optim as optim
import torchvision
from albumentations.pytorch import ToTensorV2
from shapely.wkt import loads as wkt_loads
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from PIL import Image

from solution.building_detection.unet import UNet

from .dstl_constants import *
from .dstl_processing import DstlProcessingLib


IMAGE_RES_X = 512
IMAGE_RES_Y = 512


class DstlDataset(Dataset):
    def __init__(self, transform) -> None:
        super().__init__()

        df_wkt = pd.read_csv(TRAIN_WKT_FILE)
        df_gs = pd.read_csv(GRID_SIZES_FILE)

        df_gs.rename(columns={'Unnamed: 0': COL_IMAGEID}, inplace=True)
        df_wkt = pd.merge(left=df_wkt, right=df_gs, on=COL_IMAGEID)

        # self.df_samples = df_wkt
        self.transform = transform

        self.processing = DstlProcessingLib(df_wkt, df_gs)

    def __len__(self):
        return len(self.df_wkt)

    def __getitem__(self, index):
        sample = self.df_wkt.iloc[index]

        image, mask = self.processing.read_image_and_mask(
            raster_size=(IMAGE_RES_Y, IMAGE_RES_X),
            image_id=sample[COL_IMAGEID],
        )
        mask = mask[0]

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def get_device():
    # find CUDA / MPS / CPU device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def run_train():
    ds = DstlDataset()
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    size = len(ds)

    # Create UNET
    model = UNet(in_channels=3, n_classes=10, bilinear=True)
    device = get_device()
    model.to(device)

    # Set model to train mode, necessary before backprop
    model.train()

    train_loss = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for batch, (X, y) in enumerate(dl):
        X, y = X.to(device), y.to(device)

        pred = model(X)  # forward
        loss = criterion(pred, y)  # prediction error / loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


class DstlTrain:
    def __init__(self) -> None:
        self.name = "segment"
        self.location = "models/dstl"
        self.num_epochs = 20

        self.resize_width = 512
        self.resize_height = 512

        self.device = get_device()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.model = UNet(in_channels=3, n_classes=1, bilinear=True)
        self.logits_to_probs = nn.Sigmoid()
        self.optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=0.01,
                momentum=0.9,
            )

        self.train_transform = A.Compose([
                A.Resize(height=self.resize_height, width=self.resize_width),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ])

        self.val_transform = A.Compose([
                A.Resize(height=self.resize_height, width=self.resize_width),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ])

        segm_dataset_train = DstlDataset(transform=self.train_transform)

        segm_dataset_val = DstlDataset(transform=self.val_transform)

        self.train_loader = DataLoader(segm_dataset_train, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(segm_dataset_val, batch_size=8, shuffle=False)

    def train_epoch(self):
        print("Started train one epoch.")
        num_batches = len(self.train_loader)

        self.model.train()  # set model to train mode

        train_loss = 0
        for batch_index, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device).unsqueeze(1)
            loss = self.backprop(X, y)
            train_loss += loss.item()

        train_loss /= num_batches
        print(f"Done train one epoch; AvgLoss: {train_loss}.")
        return train_loss

    def stats(self, dataset_name, data_loader):
        print("Started validation.")
        num_batches = len(data_loader)
        loss = 0.0
        error_rate = 1.0
        num_errors = 0
        num_pixels = 0

        self.model.eval() # set model to evaluation mode
        with torch.no_grad():
            for batch_index, (X, y) in enumerate(data_loader):
                X, y = X.to(self.device), y.to(self.device).unsqueeze(1)
                logits = self.forward(X)
                loss += self.criterion(logits, y).item()

                preds = self.logits_to_probs(logits)
                preds = (preds > 0.5).float()
                num_errors += (preds != y).sum()
                num_pixels += torch.numel(preds)

        loss /= num_batches

        error_rate = num_errors / num_pixels

        print(f"Stats {dataset_name}: \n ErrorRate: {(100 * error_rate):>0.1f}%, AvgLoss: {loss:>8f} \n")

        return error_rate, loss

    def forward(self, X):
        logits = self.model(X)
        return logits

    def backprop(self, X, y):
        logits = self.forward(X)
        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_predictions_as_imgs(self, data_loader, folder="saved_images"):
        self.model.eval()

        for batch_index, (X, y) in enumerate(data_loader):
            X = X.to(self.device)

            with torch.no_grad():
                logits = self.forward(X)
                preds = self.logits_to_probs(logits)
                preds = (preds > 0.5).float()

            torchvision.utils.save_image(
                preds, f"{self.location}/{folder}/pred_{batch_index}.jpg"
            )
            torchvision.utils.save_image(
                y.unsqueeze(1), f"{self.location}/{folder}/target_{batch_index}.jpg"
            )

    def train(self):
        print("Train start.")

        self.model.to(self.device)

        min_error_rate = 1.0

        logs_file = os.path.join(self.location, f"{self.name}_stats.tsv")
        f = open(logs_file, "w")
        f.write("epoch\ttrain_loss\tval_loss\ttrain_error_rate\tval_error_rate\n")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss = self.train_epoch()
            train_error_rate, train_loss = self.stats("train", self.train_loader)
            val_error_rate, val_loss = self.stats("val", self.val_loader)

            if val_error_rate < min_error_rate:
                min_error_rate = val_error_rate
                model_file = os.path.join(self.location, f"{self.name}.pt")
                torch.save(self.model, model_file)  # save best
                self.save_predictions_as_imgs(self.val_loader)

            f.write(f"{epoch}\t{train_loss}\t{val_loss}\t{train_error_rate}\t{val_error_rate}\n")
            f.flush()
        f.close()

        print("Train end.")


if __name__ == "__main__":
    print(sys.argv)

    if len(sys.argv) != 2:
        print("Please provide path to config in YAML format.")
        sys.exit(0)

    train = DstlTrain(configpath=sys.argv[1])
    train.train()
