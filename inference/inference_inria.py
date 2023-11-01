import numpy as np
import torch
from PIL import Image
from torch import nn
from os.path import basename, join
import cv2

from train.image_utils.image_gray import (
    grayscale_resize_nearest_uint8,
    probability_to_black_and_white_uint8,
    gray_nearest_black_and_white_uint8,
)
from train.image_utils.image_io import image_read, image_show, image_save
from train.segmentation.dataset_inria import VAL_TRANSFORMS

# "/Users/cristianion/Desktop/img_sat_model/inria/sample_color.jpg"
SAMPLE_PATH = (
    # "/Users/cristianion/Desktop/img_sat_model/austin1.tif"
    "/Users/cristianion/Desktop/img_sat_model/inria/sample_color.jpg"
)
OUT_PATH = "/Users/cristianion/Desktop/img_sat_model/inria/sample_color_out.png"
MODEL_PATH = "/Users/cristianion/Desktop/img_sat_model/models/inria/inria_model_1_0_4.pt"


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


class InferenceInria:
    """Segment buildings"""
    def __init__(self, debug=False, save_out=False, dir_out=None) -> None:
        self.nn_sigmoid = nn.Sigmoid()
        self.model = None
        self._load_model()
        self._debug = debug
        self._save_out = save_out
        self._dir_out = dir_out

    def _load_model(self):
        assert self.model is None
        self.model = torch.load(MODEL_PATH)
        self.model.eval()

    def image_segment_filelist(self, filelist):
        for file in filelist:
            self.image_segment_file(file)

    def image_segment_file(self, filepath):
        image = np.array(Image.open(filepath).convert("RGB"))
        if self._debug:
            image_show(image)
        segm = self.image_segment(image)
        print(segm.shape)
        if self._save_out:
            if self._dir_out:
                name = basename(filepath)
                image_save(join(self._dir_out, f"{name}.out.png"), segm)
            else:
                image_save(f"{filepath}.out.png", segm)
        if self._debug:
            image_show(segm, "segm")
        return segm

    def infer(self, img):
        img = VAL_TRANSFORMS(image=img)["image"]
        img = img[None, :]
        img = img.to(get_device())

        with torch.no_grad():
            prob = self.model(img)
            prob = self.nn_sigmoid(prob)

        print(prob.shape)
        prob = prob.to("cpu")
        prob = np.array(prob)[0][0]
        print(prob.shape)
        return prob

    def image_segment(self, image):
        print(image.shape)

        h, w, _ = image.shape
        CROP_HEIGHT = 1024
        CROP_WIDTH = 1024

        out = np.zeros((h, w), dtype=np.uint32)
        out_sigmoid = np.zeros((h, w), dtype=np.float32)

        # img_resized = image
        # pred = self.infer(img_resized)
        # pred_prob = (1.0 - 2*np.abs(0.5 - pred))
        # pred = np.where(pred > 0.5, 255, 0).astype(np.uint8)
        # # pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        # if self._debug:
        #     image_show(pred, "resized 1")
        #     pred_prob = np.clip(pred_prob, a_min=0, a_max=1.0) * 255.0
        #     pred_prob = np.round(pred_prob).astype(np.uint8)
        #     pred_prob = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        #     pred_prob = cv2.applyColorMap(pred_prob, cv2.COLORMAP_JET)
        #     image_show(pred_prob, "resized prob 1")

        # img_resized = cv2.resize(image, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_LINEAR)
        # pred = self.infer(img_resized)
        # pred_prob = (1.0 - 2*np.abs(0.5 - pred))
        # pred = np.where(pred > 0.5, 255, 0).astype(np.uint8)
        # pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        # if self._debug:
        #     image_show(pred, "resized")
        #     pred_prob = np.clip(pred_prob, a_min=0, a_max=1.0) * 255.0
        #     pred_prob = np.round(pred_prob).astype(np.uint8)
        #     pred_prob = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        #     pred_prob = cv2.applyColorMap(pred_prob, cv2.COLORMAP_JET)
        #     image_show(pred_prob, "resized prob")

        for y in range(0, h, CROP_HEIGHT):
            for x in range(0, w, CROP_WIDTH):
                crop = image[y:(y+CROP_HEIGHT), x:(x+CROP_WIDTH)]
                pred = self.infer(crop)
                out_sigmoid[y:(y+CROP_HEIGHT), x:(x+CROP_WIDTH)] = (1.0 - 2*np.abs(0.5 - pred))
                pred = np.where(pred > 0.5, 255, 0).astype(np.uint32)
                out[y:(y+CROP_HEIGHT), x:(x+CROP_WIDTH)] += pred

        if self._debug:
            out_sigmoid = np.clip(out_sigmoid, a_min=0, a_max=1.0) * 255.0
            out_sigmoid = np.round(out_sigmoid).astype(np.uint8)
            out_sigmoid = cv2.applyColorMap(out_sigmoid, cv2.COLORMAP_JET)
            image_save("sigmoid.png", out_sigmoid)

        out[:, :] = np.clip(out, a_min=0, a_max=255)
        return out.astype(np.uint8)

    def _resize(self, mask, new_w, new_h):
        return grayscale_resize_nearest_uint8(mask, new_w=new_w, new_h=new_h)

    def _threshold(self, pred):
        return probability_to_black_and_white_uint8(pred)

    def _threshold_2(self, pred):
        return gray_nearest_black_and_white_uint8(pred)


if __name__ == "__main__":
    inference = InferenceInria(debug=True, save_out=True)
    inference.image_segment_file(SAMPLE_PATH)
