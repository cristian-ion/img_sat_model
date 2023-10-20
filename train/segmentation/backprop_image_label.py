import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from train.segmentation.dstl.dataset_dstl import DSTL_NAMECODE, DstlTrainConfig
from train.segmentation.inria.dataset_inria import (
    INRIA_NAMECODE,
    InriaTrainConfig,
)
from train.segmentation.mu_buildings.dataset_mu_buildings import (
    MU_BUILDINGS_NAMECODE,
    MUBuildingsTrainConfig,
)
from train.segmentation.convnet.unet import UNet

NUM_EPOCHS = 20

VALIDATION_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_error_rate",
    "val_error_rate",
]


def draw_things(img, masks, draw_masks=True, draw_boxes=False):
    """
    Known problems:
    File "/Users/cristianion/Desktop/visual_recognition_train/train/segmentation/train_val.py", line 279, in <listcomp>
        draw_things(img, tmp)
    File "/Users/cristianion/Desktop/visual_recognition_train/train/segmentation/train_val.py", line 43, in draw_things
        boxes=masks_to_boxes(mask),
            ^^^^^^^^^^^^^^^^^^^^
    File "/Users/cristianion/Desktop/visual_recognition_train/.venv/lib/python3.11/site-packages/torchvision/ops/boxes.py", line 412, in masks_to_boxes
        bounding_boxes[index, 0] = torch.min(x)
                                ^^^^^^^^^^^^
    """
    print(f"img.shape={img.shape}")
    print(f"masks.shape={masks.shape}")
    print(masks.numel())
    canvas = img
    if draw_masks:
        canvas = draw_segmentation_masks(
            image=canvas, masks=masks, alpha=0.7, colors="red"
        )
    if draw_boxes:
        canvas = draw_bounding_boxes(
            image=canvas,
            boxes=masks_to_boxes(masks),
            colors="red",
        )
    return canvas


def plot_img(imgs, fname):
    #
    # https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#semantic-segmentation-models
    #
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
    #
    # find CUDA / MPS / CPU device
    #
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"  #
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def gen_model_id(namecode: str, major_version: int = 1, out_dir: str = None):
    if out_dir:
        files = os.listdir(out_dir)
        files = [f.split(".")[0] for f in files if f[-3:] == ".pt"]
        versions = [tuple(m.split("_")[-3:]) for m in files]
        versions = [tuple(map(int, v)) for v in versions]
        versions.sort(key=lambda x: (x[0], x[1], x[2]))

        if not versions:
            latest = (-1, 0, 0)
        else:
            latest = versions[-1]

        next_version = (0, 0, 0)
        if latest[0] > major_version:
            print("Please increase the major version constant manually.")
            raise Exception("Please increase the major version constant manually.")
        if latest[0] < major_version:
            print(f"New major version {major_version}")
            next_version = (major_version, 0, 0)
        else:
            minor = latest[1]
            subminor = latest[2] + 1
            if subminor > 9:
                subminor = 0
                minor += 1
            if minor > 9:
                raise Exception(
                    "Minor version > 9, please increase major version constant manually."
                )
            next_version = (latest[0], minor, subminor)

        major_version = next_version[0]
        minor_version = f"{next_version[1]}_{next_version[2]}"
    else:
        minor_version = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    return f"{namecode}_model_{major_version}_{minor_version}"


def train_config_by_namecode(dataset_namecode: str):
    if dataset_namecode == DSTL_NAMECODE:
        return DstlTrainConfig()
    if dataset_namecode == MU_BUILDINGS_NAMECODE:
        return MUBuildingsTrainConfig()
    if dataset_namecode == INRIA_NAMECODE:
        return InriaTrainConfig()


UNSQUEEZE_GT_ACTIVATED = [MU_BUILDINGS_NAMECODE, INRIA_NAMECODE]


class BackpropImageLabel:
    """
    Trains a models for image labeling.
    class = template of common attributes,properties (car, building, airplane, etc.)
    object = instance of the class (a specific car, a specific building, etc.)
    Applications:
    - semantic segmentation (pixel class labeling)
    - instance segmentation (pixel object labeling)
    - object detection
    - image labeling (image classification)
    """
    def __init__(self, dataset_namecode: str) -> None:
        self.device = self._get_device()
        self.dataset_namecode = dataset_namecode
        train_config = train_config_by_namecode(dataset_namecode)

        self.criterion = train_config.criterion

        self.train_loader = DataLoader(
            train_config.trainset, batch_size=train_config.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            train_config.valset,
            batch_size=train_config.val_batch_size,
            shuffle=False,
        )
        self.model = UNet(
            in_channels=3, n_classes=train_config.num_classes, bilinear=True
        )
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=0.01,
            momentum=0.9,
        )
        self.sigmoid_op = nn.Sigmoid()

        self.out_dir = f"models/{dataset_namecode}"
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        self.model_id = gen_model_id(
            train_config.namecode,
            major_version=train_config.major_version,
            out_dir=self.out_dir,
        )
        self.val_file = os.path.join(self.out_dir, f"{self.model_id}_val.tsv")
        self.model_file = os.path.join(self.out_dir, f"{self.model_id}.pt")
        self.min_error_rate = 1.0
        self.h_val_file = None

    @staticmethod
    def _get_device():
        return get_device()

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

    def _evaluate_dataset(
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

        self.save_predictions_as_imgs(self.val_loader, folder=f"figures_{epoch}")

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
                ],
            )
        )
        self.h_val_file.write(f"{val_row}\n")
        self.h_val_file.flush()

    def train(self):
        print("Train start.")

        self.model.to(self.device)

        self.evaluate_epoch(epoch=0)

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"Epoch {epoch}\n-------------------------------")
            self.train_epoch(epoch=epoch)
            self.evaluate_epoch(epoch=epoch)

        self.h_val_file.close()

        print("Train end.")


# References
# Polygonal Building Segmentation by Frame Field Learning
# https://arxiv.org/abs/2004.14875
# https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/tree/master#inria-aerial-image-labeling-dataset
#
# Todo:
#    Train the model with edges as ground truth and adjust the loss function
#
