import random
from timeit import default_timer as timer

import albumentations as A
import numpy as np
import tifffile
import torch
import torchvision
from albumentations.pytorch import ToTensorV2

from .dstl_constants import CLASSES

IMAGE_RES_Y = 512
IMAGE_RES_X = 512


class DstlEvaluate:
    def __init__(self, model_path) -> None:
        self.path = model_path
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

    def evaluate_image_filelist(self):
        pass

    def jaccard(self):
        pass


if __name__ == "__main__":
    print("DSTL evaluation")
    eval = DstlEvaluate()
    eval.evaluate_submission()


