import numpy as np
import torch
from PIL import Image
from torch import nn

from solution.image_utils.image_gray import (
    grayscale_resize_nearest_uint8,
    probability_to_black_and_white_uint8,
    gray_nearest_black_and_white_uint8,
)
from solution.image_utils.image_io import read_image, show_image
from solution.semantic_segmentation.dataset_inria import VAL_TRANSFORMS

SAMPLE_PATH = (
    "/Users/cristianion/Desktop/visual_recognition_train/inria/sample_color.jpg"
)
MODEL_PATH = "/Users/cristianion/Desktop/visual_recognition_train/models/inria/inria_model_1_0_0.pt"


def get_device():
    # find CUDA / MPS / CPU device
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


class InriaDetector:
    def __init__(self) -> None:
        self.model = load_eval_model()
        self.sigmoid = nn.Sigmoid()

    def detect_image_file(self, filepath):
        image = np.array(Image.open(filepath).convert("RGB"))
        show_image(image, "image")
        object_masks = self.detect_object_mask(image)
        print(object_masks.shape)
        show_image(object_masks, "mask")

    def detect_object_mask(self, image):
        orig_w = image.shape[1]
        orig_h = image.shape[0]
        print(image.shape)
        image = VAL_TRANSFORMS(image=image)["image"]
        image = image[None, :]
        image = image.to(get_device())

        with torch.no_grad():
            obj_mask = self.model(image)
            obj_mask = self.sigmoid(obj_mask)

        print(obj_mask.shape)
        obj_mask = obj_mask.to("cpu")
        obj_mask = np.array(obj_mask)[0][0]
        print(obj_mask.shape)

        obj_mask = probability_to_black_and_white_uint8(obj_mask)
        obj_mask = grayscale_resize_nearest_uint8(obj_mask, new_w=orig_w, new_h=orig_h)
        obj_mask = gray_nearest_black_and_white_uint8(obj_mask)
        print(obj_mask.shape)

        return obj_mask


if __name__ == "__main__":
    detector = InriaDetector()
    detector.detect_image_file(SAMPLE_PATH)
