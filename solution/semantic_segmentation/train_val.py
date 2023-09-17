import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import DataLoader

from solution.semantic_segmentation.dataset.dataset_dstl import (
    DSTL_NAMECODE,
    DstlTrainValData,
)
from solution.semantic_segmentation.dataset.dataset_mu_buildings import (
    MU_BUILDINGS_NAMECODE,
    MUBTrainValData,
)
from solution.semantic_segmentation.model.model_unet import UNet

NUM_EPOCHS = 20


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


def select_train_val_data(dataset_namecode: str):
    if dataset_namecode == DSTL_NAMECODE:
        return DstlTrainValData()
    if dataset_namecode == MU_BUILDINGS_NAMECODE:
        return MUBTrainValData()


class SemanticSegmentationTrainVal:
    def __init__(self, dataset_namecode: str) -> None:
        self.location = "segmentation/models/v1"

        self.device = get_device()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.logits_to_probs = nn.Sigmoid()
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=0.01,
            momentum=0.9,
        )

        train_val_data = select_train_val_data(dataset_namecode)

        self.train_loader = DataLoader(
            train_val_data.trainset, batch_size=train_val_data.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            train_val_data.valset, batch_size=train_val_data.batch_size, shuffle=False
        )
        self.model = UNet(
            in_channels=3, n_classes=train_val_data.num_classes, bilinear=True
        )
        self.name = gen_model_id(train_val_data.namecode, version=1)

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

        self.model.eval()  # set model to evaluation mode
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

        print(
            f"Stats {dataset_name}: \n ErrorRate: {(100 * error_rate):>0.1f}%, AvgLoss: {loss:>8f} \n"
        )

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

    def train_val(self):
        print("Train start.")

        self.model.to(self.device)

        min_error_rate = 1.0

        logs_file = os.path.join(self.location, f"{self.name}_stats.tsv")
        f = open(logs_file, "w")
        f.write("epoch\ttrain_loss\tval_loss\ttrain_error_rate\tval_error_rate\n")
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss = self.train_epoch()
            train_error_rate, train_loss = self.stats("train", self.train_loader)
            val_error_rate, val_loss = self.stats("val", self.val_loader)

            if val_error_rate < min_error_rate:
                min_error_rate = val_error_rate
                model_file = os.path.join(self.location, f"{self.name}.pt")
                torch.save(self.model, model_file)  # save best
                self.save_predictions_as_imgs(self.val_loader)

            f.write(
                f"{epoch}\t{train_loss}\t{val_loss}\t{train_error_rate}\t{val_error_rate}\n"
            )
            f.flush()
        f.close()

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
