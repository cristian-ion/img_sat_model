import numpy as np
import torch
from PIL import Image
from torch import nn

from solution.semantic_segmentation.dataset_inria import VAL_TRANSFORMS
from solution.semantic_segmentation.train_val import draw_things, plot_img
from solution.image_utils.image_gray import binarize_grayscale, grayscale_resize
from solution.image_utils.image_io import show_image
import torchvision.transforms.functional as F


SAMPLE_PATH = (
    "/Users/cristianion/Desktop/visual_recognition_train/inria/sample_color.jpg"
)
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
    orig_w = image.shape[1]
    orig_h = image.shape[0]
    print(image.shape)

    image = VAL_TRANSFORMS(image=image)["image"]
    image = image[None, :]
    image = image.to(get_device())

    with torch.no_grad():
        preds = model(image)
        masks = sigmoid(preds)
    # masks = masks > 0.5
    # print(masks.shape)

    # image = image.to("cpu")
    # masks = masks.to("cpu")

    # img_arr = np.array(Image.open(image_path).convert("RGB"))
    # img_arr = np.moveaxis(img_arr, -1, 0)
    # image = [
    #     torch.tensor(img_arr)
    # ]

    # drawn_masks_and_boxes = [
    #     draw_things(img, tmp) for img, tmp in zip(image, masks)
    # ]

    # show_image(drawn_masks_and_boxes, title="window")

    masks = masks.to("cpu")
    out = np.array(masks)[0][0]
    print(out.shape)

    out = np.where(out > 0.5, 255, 0).astype(np.uint8)
    out = grayscale_resize(out, new_w=orig_w, new_h=orig_h)
    out = np.where(out > 127, 255, 0).astype(np.uint8)
    print(out.shape)
    show_image(out)

    return masks




if __name__ == "__main__":
    masks = predict_one(SAMPLE_PATH)