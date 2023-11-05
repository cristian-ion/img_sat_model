import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD
from time import time

from train.helpers.gen_model_id import gen_model_id
from train.helpers.get_device import get_device
from train.image_utils.segm_utils import draw_things, plot_img
from train.segmentation import (
    INRIA_NAMECODE,
    MU_BUILDINGS_NAMECODE,
    train_config_by_namecode,
)
from train.segmentation.convnet.unet import UNet
from train.segmentation.convnet.unet_valid import UNetValid
from train.helpers.early_stopper import EarlyStopper

DEFAULT_NUM_EPOCHS = 25
VALIDATION_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_error_rate",
    "val_error_rate",
]
UNSQUEEZE_GT_ACTIVATED = [MU_BUILDINGS_NAMECODE, INRIA_NAMECODE]


class Train:
    """
    Trains a models for image labeling.
    class = template of common attributes,properties (car, building, airplane, etc.)
    object = instance of the class (a specific car, a specific building, etc.)
    Applications:
    - semantic segmentation (pixel class labeling)
    - instance segmentation (pixel object labeling)
    - object detection
    - image labeling (image classification)
    ...
    This class will later replace the classification class, so we will do classification also here.
    """

    def __init__(self, dataset_namecode: str, checkpoint=None) -> None:
        self.device = self._get_device()
        self.dataset_namecode = dataset_namecode
        train_config = train_config_by_namecode(dataset_namecode)

        self.criterion = train_config.criterion
        self.num_epochs = train_config.num_epochs or DEFAULT_NUM_EPOCHS

        self.train_loader = DataLoader(
            train_config.trainset,
            batch_size=train_config.batch_size,
            shuffle=True,
            prefetch_factor=1,
            num_workers=1,
        )
        self.val_loader = DataLoader(
            train_config.valset,
            batch_size=train_config.val_batch_size,
            shuffle=False,
        )

        if not checkpoint:
            print("No checkpoint")

        self.model = UNetValid(
            in_channels=3, n_classes=train_config.num_classes, bilinear=True
        )
        # optimizer settings
        self.optimizer = SGD(
            params=self.model.parameters(),
            lr=0.01,
            momentum=0.9,
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=5)
        self.early_stopper = EarlyStopper(patience=3, min_delta=1)

        # activation unit
        self.sigmoid_op = nn.Sigmoid()
        self.out_dir = f"models/{dataset_namecode}"
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        self.model_id = gen_model_id(
            train_config.namecode,
            major_version=train_config.major_version,
            out_dir=self.out_dir,
        )
        self.val_file_path = os.path.join(self.out_dir, f"{self.model_id}_val.tsv")
        self.model_file = os.path.join(self.out_dir, f"{self.model_id}.pt")
        self.model_checkpoint_file = os.path.join(self.out_dir, f"{self.model_id}.cp")
        self.min_error_rate = 1.0
        self.val_file_handle = None

    def train(self):
        print("Train start.")

        self.model.to(self.device)

        self.val_file_handle = open(self.val_file_path, "w")
        val_file_header = "\t".join(VALIDATION_COLUMNS)
        self.val_file_handle.write(f"{val_file_header}\n")
        self.val_file_handle.flush()

        # self.validate_epoch(epoch=0, train_loss=1.0)

        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}\n-------------------------------")
            train_loss = self.train_epoch(epoch=epoch)
            val_loss = self.validate_epoch(epoch=epoch, train_loss=train_loss)
            self._save_min_val_error_rate(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            self.scheduler.step(val_loss)
            if self.early_stopper.step(val_loss):
                break

        self.val_file_handle.close()
        print("Train end.")

    @staticmethod
    def _get_device():
        return get_device()

    def train_epoch(self, epoch: int):
        print(f"Started train epoch {epoch}.")

        num_batches = len(self.train_loader)
        self.model.train()  # set model to train mode

        train_loss = 0
        for batch_index, (X, y) in enumerate(self.train_loader):
            ts = time()
            X, y = X.to(self.device), y.to(self.device)

            if self.dataset_namecode in UNSQUEEZE_GT_ACTIVATED:
                y = y.unsqueeze(1)

            loss = self.backprop(X, y)
            train_loss += loss.item()
            te = time() - ts
            print(f"{batch_index} / {num_batches}, duration: {te}s")

        train_loss /= num_batches
        print(f"Done train epoch {epoch}; AvgLoss: {train_loss}.")
        return train_loss

    def _torch_binarize(self, preds):
        return (preds > 0.5).float()

    def _torch_not_equal(self, preds, target):
        return preds != target

    def _validate_on_dataset(
        self, dataset_name: str, data_loader: DataLoader
    ) -> tuple[float, float]:
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

    def save_predictions_as_imgs(
        self, data_loader, separate_mask=False, folder="saved_images"
    ):
        #
        # https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#semantic-segmentation-models
        # https://pytorch.org/vision/main/auto_examples/others/plot_repurposing_annotations.html#
        #
        if self.dataset_namecode not in UNSQUEEZE_GT_ACTIVATED:
            return

        self.model.eval()

        if not os.path.exists(f"{self.out_dir}/{folder}"):
            os.mkdir(f"{self.out_dir}/{folder}")

        for batch_index, (X, gt) in enumerate(data_loader):
            X = X.to(self.device)

            with torch.no_grad():
                logits = self.forward(X)
                mask = self.sigmoid_op(logits)
                mask = mask > 0.5

            if separate_mask:
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(mask),
                    f"{self.out_dir}/{folder}/mask_{batch_index}.jpg",
                )
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(gt.unsqueeze(1)),
                    f"{self.out_dir}/{folder}/gt_{batch_index}.jpg",
                )
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(X),
                    f"{self.out_dir}/{folder}/image_{batch_index}.jpg",
                )
            else:
                X = X.to("cpu")
                mask = mask.to("cpu")
                X = F.convert_image_dtype(X, torch.uint8)

                drawn_masks_and_boxes = [
                    draw_things(img, tmp) for img, tmp in zip(X, mask)
                ]

                plot_img(
                    drawn_masks_and_boxes,
                    f"{self.out_dir}/{folder}/mask_{batch_index}.jpg",
                )
            break

    def _save_min_val_error_rate(self, epoch, train_loss, val_loss):
        if val_loss < self.min_error_rate:
            self.min_error_rate = val_loss
            state = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'early_stopper': self.early_stopper.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }

            # checkpoint
            torch.save(state, self.model_checkpoint_file)

            # full model
            torch.save(self.model, self.model_file)

    def validate_epoch(self, epoch: int, train_loss: float = None):
        # if epoch % 10 == 0:
        #     self.save_predictions_as_imgs(self.val_loader, folder=f"figures_{epoch}")

        print("Started epoch validation.")
        if not train_loss:
            train_error_rate, train_loss = self._validate_on_dataset(
                "train", self.train_loader
            )
        else:
            train_error_rate = "nan"
        val_error_rate, val_loss = self._validate_on_dataset("val", self.val_loader)
        print("Ended epoch validation.")

        val_row = "\t".join(
            map(
                str,
                [
                    epoch,
                    train_loss,
                    val_loss,
                    train_error_rate,
                    val_error_rate,
                ],
            )
        )
        self.val_file_handle.write(f"{val_row}\n")
        self.val_file_handle.flush()
        return val_loss



# References
# Polygonal Building Segmentation by Frame Field Learning
# https://arxiv.org/abs/2004.14875
# https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/tree/master#inria-aerial-image-labeling-dataset
#
# Todo:
#    Train the model with edges as ground truth and adjust the loss function
#
