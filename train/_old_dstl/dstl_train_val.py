"""
Implementation reference
- https://www.kaggle.com/code/alijs1/squeezed-this-in-successful-kernel-run
"""

import os
from datetime import datetime

import albumentations as A
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader

from train.segmentation.convnet.unet import UNet

from ..segmentation.dstl.dataset_dstl import DstlDataset
from .dstl_constants import (
    CLASSES,
    GRID_SIZES_FILE,
    IMAGE_RES_X,
    IMAGE_RES_Y,
    TRAIN_WKT_FILE,
)


class DstlTrain:
    def __init__(self) -> None:
        date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.version = 1
        self.unique_id = f"dstl_model_{self.version}_{date_time}"

        self.out_path = "models/dstl"
        self.num_epochs = 20
        self.batch_size = 8
        self.device = self.get_device()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.model = UNet(in_channels=3, n_classes=len(CLASSES), bilinear=True)
        self.logits_to_probs = nn.Sigmoid()
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=0.01,
            momentum=0.9,
        )

        DSTL_TRAIN_TRANSFORM = A.Compose(
            [
                A.Resize(height=IMAGE_RES_Y, width=IMAGE_RES_X),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        DSTL_VAL_TRANSFORM = A.Compose(
            [
                A.Resize(height=IMAGE_RES_Y, width=IMAGE_RES_X),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        dstl_trainset = DstlDataset(
            DSTL_TRAIN_TRANSFORM,
            train_csv=TRAIN_WKT_FILE,
            grid_csv=GRID_SIZES_FILE,
            classes=CLASSES,
            train_res_x=IMAGE_RES_X,
            train_res_y=IMAGE_RES_Y,
        )
        dstl_valset = DstlDataset(
            DSTL_VAL_TRANSFORM,
            train_csv=TRAIN_WKT_FILE,
            grid_csv=GRID_SIZES_FILE,
            classes=CLASSES,
            train_res_x=IMAGE_RES_X,
            train_res_y=IMAGE_RES_Y,
        )

        self.train_loader = DataLoader(
            dstl_trainset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            dstl_valset, batch_size=self.batch_size, shuffle=False
        )

    def get_device(self):
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

    def train_epoch(self):
        print("Started train one epoch.")
        num_batches = len(self.train_loader)

        self.model.train()  # set model to train mode

        train_loss = 0
        for batch_index, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            loss = self.backprop(X, y)
            train_loss += loss.item()

        train_loss /= num_batches
        print(f"Done train one epoch; AvgLoss: {train_loss}.")
        return train_loss

    def validation_epoch(self, dataset_name, data_loader) -> tuple[float, float]:
        print("Started validation.")
        num_batches = len(data_loader)
        loss = 0.0
        error_rate = 1.0
        num_errors = 0
        num_pixels = 0

        self.model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for batch_index, (X, y) in enumerate(data_loader):
                X, y = X.to(self.device), y.to(self.device)

                logits = self.forward(X)

                loss += self.criterion(logits, y).item()

                preds = self.logits_to_probs(logits)
                preds = (preds > 0.5).float()
                num_errors += (preds != y).sum()
                num_pixels += torch.numel(preds)

        loss /= num_batches

        error_rate = num_errors / num_pixels

        print(
            f"Stats {dataset_name}: \n ErrorRate: {(100 * error_rate):>0.1f}%, AvgLoss: {loss:>8f} \n"
        )

        return error_rate, loss

    def forward(self, X):
        logits = self.model(X)
        return logits

    def backprop(self, X, y):
        logits = self.forward(X)
        print(logits.shape, y.shape)

        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_predictions_as_imgs(self, data_loader, folder="saved_images"):
        self.model.eval()

        if not os.path.exists(f"{self.out_path}/{folder}"):
            os.mkdir(f"{self.out_path}/{folder}")

        for batch_index, (X, y) in enumerate(data_loader):
            X = X.to(self.device)

            with torch.no_grad():
                logits = self.forward(X)
                preds = self.logits_to_probs(logits)
                preds = (preds > 0.5).float()

            torchvision.utils.save_image(
                preds, f"{self.out_path}/{folder}/pred_{batch_index}.jpg"
            )
            torchvision.utils.save_image(
                y.unsqueeze(1), f"{self.out_path}/{folder}/target_{batch_index}.jpg"
            )

    def train(self):
        print("Train start.")

        self.model.to(self.device)

        min_error_rate = 1.0

        logs_file = os.path.join(self.out_path, f"{self.unique_id}_stats.tsv")
        f = open(logs_file, "w")
        f.write("epoch\ttrain_loss\tval_loss\ttrain_error_rate\tval_error_rate\n")
        train_error_rate, train_loss = self.validation_epoch("train", self.train_loader)
        val_error_rate, val_loss = self.validation_epoch("val", self.val_loader)

        epoch = -1
        f.write(
            f"{epoch}\t{train_loss}\t{val_loss}\t{train_error_rate}\t{val_error_rate}\n"
        )
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss = self.train_epoch()
            train_error_rate, train_loss = self.validation_epoch(
                "train", self.train_loader
            )
            val_error_rate, val_loss = self.validation_epoch("val", self.val_loader)

            if val_error_rate < min_error_rate:
                min_error_rate = val_error_rate
                model_file = os.path.join(self.out_path, f"{self.unique_id}.pt")
                torch.save(self.model, model_file)  # save best

            f.write(
                f"{epoch}\t{train_loss}\t{val_loss}\t{train_error_rate}\t{val_error_rate}\n"
            )
            f.flush()
        f.close()

        print("Train end.")


def train():
    dstl_train = DstlTrain()
    dstl_train.train()


if __name__ == "__main__":
    train()
