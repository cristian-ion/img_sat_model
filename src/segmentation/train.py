from segmentation.unet import UNet
from segmentation.dataset import SegmentationDataset

import os
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision


IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


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



class ImgSegmentTrain():
    def __init__(self) -> None:
        self.name = "segment"
        self.location = "segmentation/models/v1"
        self.num_epochs = 20

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
            ])

        self.val_transform = A.Compose([
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ])

        segm_dataset_train = SegmentationDataset(
            image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train",
            mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/train_labels",
            transform=self.train_transform)

        segm_dataset_val = SegmentationDataset(
            image_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val",
            mask_dir="/Users/cristianion/Desktop/satimg_data/Massachusetts Buildings Dataset/png/val_labels",
            transform=self.val_transform)

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


def main():
    trainer = ImgSegmentTrain()
    trainer.train()


if __name__ == "__main__":
    main()
