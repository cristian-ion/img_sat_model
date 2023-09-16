import torch
import tifffile
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision
from timeit import default_timer as timer
import random

from .dstl_constants import CLASSES

IMAGE_RES_Y = 512
IMAGE_RES_X = 512


class DstlEvaluate:
    def __init__(self, path) -> None:
        self.path = path
        self.model = torch.load(self.path)
        self.device = self.get_device()
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(height=IMAGE_RES_Y, width=IMAGE_RES_X),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

        self.sigmoid = torch.nn.Sigmoid()

    def evaluate(self):
        pass

    def jaccard_index(self):
        pass



if __name__ == "__main__":
    print("DSTL evaluation")
    eval = DstlEvaluate()
    eval.evaluate()


