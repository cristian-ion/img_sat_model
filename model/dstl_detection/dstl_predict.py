import os

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



class DstlPredict:
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

    def predict(self, image_path):
        img = tifffile.imread(image_path)
        original_img_size = img.shape
        if img.shape[0] == 3:
            img = np.rollaxis(img, 0, 3)

        preprocessing = self.transform(image=img)
        img = preprocessing["image"]
        img = img.unsqueeze(0)

        img = img.to(self.device)

        with torch.no_grad():
            logits = self.model(img)

        # resize back to original image size;
        print(original_img_size)

        probs = self.sigmoid(logits)
        mask = (probs > 0.5).float()

        return mask


def test_predict():
    model_path = "/Users/cristianion/Desktop/satimg_model/models/dstl/dstl_model_1_2023_09_08_10_15_26.pt"
    image_path = "/Users/cristianion/Desktop/satimg_model/samples/sample_dstl.tif"

    dstl_predict = DstlPredict(path=model_path)

    result = dstl_predict.predict(image_path=image_path)

    result = result.squeeze(0)

    print(result)
    print(result.shape)
    print(result.min())
    print(result.max())

    for i in range(len(CLASSES)):
        torchvision.utils.save_image(result[i], f"samples/sample_dstl_{CLASSES[i]}.png")


def test_multiple_predict():
    model_id = "dstl_model_1_2023_09_08_10_15_26"
    model_path = "/Users/cristianion/Desktop/satimg_model/models/dstl/dstl_model_1_2023_09_08_10_15_26.pt"
    path = "/Users/cristianion/Desktop/satimg_data/DSTL/three_band"

    files = [(os.path.join(path, file), file) for file in os.listdir(path) if file.split('.')[-1] == 'tif']

    files = random.sample(files, k=int(0.1 * float(len(files))))

    dstl_predict = DstlPredict(path=model_path)

    times = []
    for file, name in files:
        print(f"Scanning {file}")
        start = timer()
        result = dstl_predict.predict(file)
        result = result.squeeze(0)
        end = timer()
        duration = end-start
        print(f"Predict time {duration}")
        times.append(duration)

        for i in range(len(CLASSES)):
            torchvision.utils.save_image(result[i], f"/Users/cristianion/Documents/out/{CLASSES[i]}_{model_id}_{name}.png")

    print(min(times))
    print(max(times))


if __name__ == "__main__":
    test_predict()
    test_multiple_predict()