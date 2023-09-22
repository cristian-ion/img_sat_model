import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from solution.semantic_segmentation.dataset_dstl import DSTL_NAMECODE, DstlTrainValData
from solution.semantic_segmentation.dataset_mu_buildings import (
    MU_BUILDINGS_NAMECODE,
    MUBTrainValData,
)
from solution.semantic_segmentation.dataset_inria import (
    INRIA_NAMECODE,
    InriaTrainValData,
)
from solution.semantic_segmentation.model_unet import UNet

import torchvision.transforms.functional as F

NUM_EPOCHS = 20

VALIDATION_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_error_rate",
    "val_error_rate",
]

def show(imgs, fname):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(fname=fname, dpi=400)

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


def gen_model_id(name, version=1):
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{name}_model_{version}_{date_time}"


def train_val_data_factory(dataset_namecode: str):
    if dataset_namecode == DSTL_NAMECODE:
        return DstlTrainValData()
    if dataset_namecode == MU_BUILDINGS_NAMECODE:
        return MUBTrainValData()
    if dataset_namecode == INRIA_NAMECODE:
        return InriaTrainValData()


UNSQUEEZE_GT_ACTIVATED = [MU_BUILDINGS_NAMECODE, INRIA_NAMECODE]


class SemanticSegmentationTrainVal:
    def __init__(self, dataset_namecode: str) -> None:
        self.device = get_device()
        self.dataset_namecode = dataset_namecode
        train_val_data = train_val_data_factory(dataset_namecode)

        self.criterion = train_val_data.criterion

        self.train_loader = DataLoader(
            train_val_data.trainset, batch_size=train_val_data.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            train_val_data.valset, batch_size=train_val_data.batch_size, shuffle=False
        )
        self.model = UNet(
            in_channels=3, n_classes=train_val_data.num_classes, bilinear=True
        )
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=0.01,
            momentum=0.9,
        )
        self.sigmoid_op = nn.Sigmoid()

        self.out_path = f"models/{dataset_namecode}"
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)
        self.model_id = gen_model_id(train_val_data.namecode, version=1)
        self.val_file = os.path.join(self.out_path, f"{self.model_id}_val.tsv")
        self.model_file = os.path.join(self.out_path, f"{self.model_id}.pt")
        self.min_error_rate = 1.0
        self.h_val_file = None

    def train_epoch(self, epoch: int):
        print(f"Started train epoch {epoch}.")

        num_batches = len(self.train_loader)
        self.model.train()  # set model to train mode

        train_loss = 0
        for batch_index, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            if self.dataset_namecode in UNSQUEEZE_GT_ACTIVATED:
                y = y.unsqueeze(1)

            loss = self.backprop(X, y)
            train_loss += loss.item()

        train_loss /= num_batches
        print(f"Done train epoch {epoch}; AvgLoss: {train_loss}.")

    def _torch_binarize(self, preds):
        return (preds > 0.5).float()

    def _torch_not_equal(self, preds, target):
        return preds != target

    def _evaluate_dataset(self, dataset_name: str, data_loader: DataLoader) -> tuple[float, float]:
        num_batches = len(data_loader)
        loss = 0.0
        error_rate = 1.0

        num_errors = 0.0
        num_pixels = 0.0

        self.model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for batch_index, (X, y) in enumerate(data_loader):
                X, y = X.to(self.device), y.to(self.device)

                if self.dataset_namecode in UNSQUEEZE_GT_ACTIVATED:
                    y = y.unsqueeze(1)

                logits = self.forward(X)
                loss += self.criterion(logits, y).item()

                preds = self.sigmoid_op(logits)
                preds = self._torch_binarize(preds=preds)
                num_errors += self._torch_not_equal(preds, y).sum()
                num_pixels += torch.numel(preds)

        loss /= num_batches

        error_rate = num_errors / num_pixels

        print(
            f"Evaluate {dataset_name}: \n ErrorRate: {(100 * error_rate):>0.1f}%, AvgLoss: {loss:>8f} \n"
        )

        return error_rate.item(), loss

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
        if self.dataset_namecode not in UNSQUEEZE_GT_ACTIVATED:
            return

        self.model.eval()

        if not os.path.exists(f"{self.out_path}/{folder}"):
            os.mkdir(f"{self.out_path}/{folder}")

        for batch_index, (X, gt) in enumerate(data_loader):
            X = X.to(self.device)

            with torch.no_grad():
                logits = self.forward(X)
                mask = self.sigmoid_op(logits)
                mask = (mask > 0.5)

            # 3 separate images
            if True is False:
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(mask), f"{self.out_path}/{folder}/mask_{batch_index}.jpg"
                )
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(gt.unsqueeze(1)), f"{self.out_path}/{folder}/gt_{batch_index}.jpg"
                )
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(X), f"{self.out_path}/{folder}/image_{batch_index}.jpg"
                )
            else:
                X = X.to('cpu')
                mask = mask.to('cpu')
                X = F.convert_image_dtype(X, torch.uint8)
                buildings_with_masks = [
                    torchvision.utils.draw_segmentation_masks(image=img, masks=tmp, alpha=0.7, colors='red')
                    for img, tmp in zip(X, mask)
                ]
                show(
                    buildings_with_masks,
                    f"{self.out_path}/{folder}/mask_{batch_index}.jpg",
                )

    def _save_min_val_error_rate(self, val_error_rate):
        if val_error_rate < self.min_error_rate:
            self.min_error_rate = val_error_rate
            torch.save(self.model, self.model_file)

    def evaluate_epoch(self, epoch: int):
        if epoch == 0:
            self.h_val_file = open(self.val_file, "w")
            val_file_header = "\t".join(VALIDATION_COLUMNS)
            self.h_val_file.write(f"{val_file_header}\n")
            self.h_val_file.flush()

        self.save_predictions_as_imgs(self.val_loader, folder=f"out_{epoch}")

        print("Started epoch validation.")
        train_error_rate, train_loss = self._evaluate_dataset(
            "train", self.train_loader
        )
        val_error_rate, val_loss = self._evaluate_dataset("val", self.val_loader)
        print("Ended epoch validation.")

        self._save_min_val_error_rate(val_error_rate=val_error_rate)

        val_row = "\t".join(
            map(
                str,
                [
                    epoch,
                    train_loss,
                    val_loss,
                    train_error_rate,
                    val_error_rate,
                ]
            )
        )
        self.h_val_file.write(f"{val_row}\n")
        self.h_val_file.flush()

    def train_val(self):
        print("Train start.")

        self.model.to(self.device)

        self.evaluate_epoch(epoch=0)

        for epoch in range(1, NUM_EPOCHS+1):
            print(f"Epoch {epoch}\n-------------------------------")
            self.train_epoch(epoch=epoch)
            self.evaluate_epoch(epoch=epoch)

        self.h_val_file.close()

        print("Train end.")


def main():
    print(sys.argv)

    if len(sys.argv) != 2:
        print(
            f"Please provide dataset namecode: {MU_BUILDINGS_NAMECODE} or {DSTL_NAMECODE}."
        )
        sys.exit(0)

    trainer = SemanticSegmentationTrainVal(sys.argv[1])
    trainer.train_val()


if __name__ == "__main__":
    main()
