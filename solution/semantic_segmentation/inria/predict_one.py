import torch
from PIL import Image
import numpy as np
from torch import nn

from solution.semantic_segmentation.dataset_inria import (
    VAL_TRANSFORMS
)


SAMPLE_PATH = "/Users/cristianion/Desktop/visual_recognition_train/inria/sample_color.jpg"
MODEL_PATH = "/Users/cristianion/Desktop/visual_recognition_train/models/inria/inria_model_1_0_0.pt"

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

def load_eval_model():
    model = torch.load(MODEL_PATH)
    model.eval()
    return model


def predict_one(image_path):
    model = load_eval_model()
    sigmoid = nn.Sigmoid()

    image = np.array(Image.open(image_path).convert("RGB"))

    orignal_size = image.shape

    image = VAL_TRANSFORMS(image=image)["image"]
    image = image[None, :]
    image = image.to(get_device())

    preds = model(image)
    masks = sigmoid(preds)

    print(masks.shape)


if __name__ == "__main__":
    predict_one(SAMPLE_PATH)