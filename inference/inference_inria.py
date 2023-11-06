import numpy as np
import torch
from PIL import Image
from torch import nn
from os.path import basename, join
import cv2
import math

from train.image_utils.image_gray import (
    grayscale_resize_nearest_uint8,
)
from train.image_utils.image_io import image_show, image_save
from train.segmentation.dataset_inria import A, ToTensorV2, VAL_TRANSFORMS
from constants import REPO_DIR


SAMPLE_PATH = f"{REPO_DIR}/inria/sample_color.jpg"

MODEL_1_0_7_PATH = f"{REPO_DIR}/models/inria/inria_model_1_0_7.pt"
MODEL_1_0_6_PATH = f"{REPO_DIR}/models/inria/inria_model_1_0_6.pt"
MODEL_1_0_5_PATH = f"{REPO_DIR}/models/inria/inria_model_1_0_5.pt"
MODEL_1_0_4_PATH = f"{REPO_DIR}/models/inria/inria_model_1_0_4.pt"
MODEL_1_0_3_PATH = f"{REPO_DIR}/models/inria/inria_model_1_0_3.pt"

INRIA_MODEL_1_0_7_NAME = "inria_model_1_0_7"
INRIA_MODEL_1_0_6_NAME = "inria_model_1_0_6"
INRIA_MODEL_1_0_5_NAME = "inria_model_1_0_5"
INRIA_MODEL_1_0_4_NAME = "inria_model_1_0_4"
INRIA_MODEL_1_0_3_NAME = "inria_model_1_0_3"

MODELS = {
    INRIA_MODEL_1_0_7_NAME: MODEL_1_0_7_PATH,
    INRIA_MODEL_1_0_6_NAME: MODEL_1_0_6_PATH,
    INRIA_MODEL_1_0_5_NAME: MODEL_1_0_5_PATH,
    INRIA_MODEL_1_0_4_NAME: MODEL_1_0_4_PATH,
    INRIA_MODEL_1_0_3_NAME: MODEL_1_0_3_PATH,
}
LATEST_MODEL_NAME = INRIA_MODEL_1_0_7_NAME
LATEST_MODEL_PATH = MODELS[LATEST_MODEL_NAME]

RGB_MEAN = (math.ceil(0.485 * 255), math.ceil(0.456 * 255), math.ceil(0.406 * 255))
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STDDEV = (0.229, 0.224, 0.225)
# NORM_STDDEV = (1, 1, 1)


IMAGE_TRANSFORMS = A.Compose(
    transforms=[
        A.Normalize(
            mean=NORM_MEAN,
            std=NORM_STDDEV,
            max_pixel_value=255.0,
            always_apply=True,
        ),
        ToTensorV2(),
    ],
    is_check_shapes=False,
)


def get_device():
    # find CUDA / MPS / CPU device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"  #
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # print(f"Using {device} device")
    return device


def crop_img(img, y=None, x=None, h=None, w=None, border=None):
    y = y or 0
    x = x or 0
    h = h or img.shape[0]
    w = w or img.shape[1]
    border = border or 0
    img = img[y+border:(y+h-border), x+border:(x+w-border)]
    return img


class InferenceInria:
    """Segment buildings"""
    def __init__(self, model_name=LATEST_MODEL_NAME, debug=False, save_out=False, dir_out=None, include_px_prob=False, model_path=None) -> None:
        """
        dir_out: is usefull when scanning a directory.
        save_out: saves the output in out.png file.
        """
        self.model = None
        self.nn_sigmoid = nn.Sigmoid()
        self.model_name = model_name
        if not model_path:
            model_path = MODELS[model_name]
        print(model_path)
        self._load_model(model_path)
        self._debug = debug
        self._save_out = save_out
        self._dir_out = dir_out
        self.include_px_prob = include_px_prob

    def _load_model(self, model_path):
        assert self.model is None
        self.model = torch.load(model_path)
        self.model.eval()

    def infer_filelist(self, filelist):
        for file in filelist:
            self.infer_file(file)

    def infer_file(self, filepath):
        image = cv2.imread(filepath)
        # if self._debug:
        #     image_show(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.model_name == INRIA_MODEL_1_0_4_NAME:
            px_cls, px_prob = self.image_segment_v2(image)
        else:
            px_cls, px_prob = self.image_segment_v3(image)

        if self._save_out:
            if self._dir_out:
                name = basename(filepath)
                image_save(join(self._dir_out, f"{name}.out.png"), px_cls)
            else:
                image_save(f"{filepath}.out.png", px_cls)

        if self._debug:
            image_show(px_cls, "px_cls")
            image_show(px_prob, "px_prob")
        return px_cls, px_prob

    def _infer(self, img):
        img = IMAGE_TRANSFORMS(image=img)["image"]
        img = img[None, :]
        img = img.to(get_device())

        with torch.no_grad():
            px_prob = self.model(img)
            px_prob = self.nn_sigmoid(px_prob)

        px_prob = px_prob.to("cpu")
        px_prob = np.array(px_prob)[0][0]
        return px_prob

    def image_segment(self, image):
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
                pred = self._infer(crop)
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

    def image_segment_v1(self, image):
        print(image.shape)
        img_height = image.shape[0]
        img_width = image.shape[1]
        GT_CROP_HEIGHT = 388
        IMG_CROP_HEIGHT = 572

        gt_border_size_y = abs(img_height - (math.ceil(img_height / GT_CROP_HEIGHT) * GT_CROP_HEIGHT))//2
        img_border_size_y = gt_border_size_y + (IMG_CROP_HEIGHT - GT_CROP_HEIGHT)//2
        print(gt_border_size_y)
        print(img_border_size_y)

        gt_border_size_x = abs(img_width - (math.ceil(img_width / GT_CROP_HEIGHT) * GT_CROP_HEIGHT))//2
        img_border_size_x = gt_border_size_x + (IMG_CROP_HEIGHT - GT_CROP_HEIGHT)//2
        print(gt_border_size_x)
        print(img_border_size_x)

        image = self._padding(image, border_y=img_border_size_y, border_x=img_border_size_x)
        # image_show(image, "padding")
        print(image.shape)
        out = self._infer(image)
        out = np.where(out > 0.5, 255, 0).astype(np.uint32)
        out[:, :] = np.clip(out, a_min=0, a_max=255)
        out = out[gt_border_size_y:-gt_border_size_y, gt_border_size_x:-gt_border_size_x]
        print(out.shape)
        return out.astype(np.uint8), out.astype(np.uint8)

    def image_segment_v2(self, image):
        # print(image.shape)
        img_height = image.shape[0]
        img_width = image.shape[1]
        GT_CROP_HEIGHT = 388
        IMG_CROP_HEIGHT = 572

        gt_border_size_y = abs(img_height - (math.ceil(img_height / GT_CROP_HEIGHT) * GT_CROP_HEIGHT))//2
        img_border_size_y = gt_border_size_y + (IMG_CROP_HEIGHT - GT_CROP_HEIGHT)//2
        # print(gt_border_size_y)
        # print(img_border_size_y)

        gt_border_size_x = abs(img_width - (math.ceil(img_width / GT_CROP_HEIGHT) * GT_CROP_HEIGHT))//2
        img_border_size_x = gt_border_size_x + (IMG_CROP_HEIGHT - GT_CROP_HEIGHT)//2
        # print(gt_border_size_x)
        # print(img_border_size_x)

        image = self._padding(image, border_y=img_border_size_y, border_x=img_border_size_x)
        # print(image.shape)

        height = image.shape[0]
        width = image.shape[1]

        out = np.zeros((height, width), dtype=np.uint32)

        df = (IMG_CROP_HEIGHT - GT_CROP_HEIGHT)//2

        for y in range(df, height-img_border_size_y, GT_CROP_HEIGHT):
            for x in range(df, width-img_border_size_x, GT_CROP_HEIGHT):
                crop = image[y-df:(y-df+IMG_CROP_HEIGHT), (x-df):(x-df+IMG_CROP_HEIGHT)]
                pred = self._infer(crop)
                pred = np.where(pred > 0.5, 255, 0).astype(np.uint8)
                # image_show(pred, title=f"pred {pred.shape} df {df}")
                assert pred.shape[0] == IMG_CROP_HEIGHT, pred.shape
                assert pred.shape[1] == IMG_CROP_HEIGHT, pred.shape
                pred = pred[df:(df+GT_CROP_HEIGHT), df:(df+GT_CROP_HEIGHT)]
                # image_show(pred, title=f"pred {pred.shape} df {df}")
                out[y:(y+GT_CROP_HEIGHT), x:(x+GT_CROP_HEIGHT)] = pred

        out[:, :] = np.clip(out, a_min=0, a_max=255)
        print(out.shape)
        out = out[img_border_size_y:-img_border_size_y, img_border_size_x:-img_border_size_x]
        return out.astype(np.uint8), out.astype(np.uint8)

    def image_segment_v3(self, img):
        height = img.shape[0]
        width = img.shape[1]

        STRIDE_Y = 388
        STRIDE_X = 388
        CROP_HEIGHT = 572
        CROP_WIDTH = 572
        DIFF = (CROP_HEIGHT - STRIDE_Y)//2

        rows = math.ceil(height / STRIDE_Y)
        cols = math.ceil(width / STRIDE_X)
        border_y = (rows * STRIDE_Y - height)//2 + DIFF
        border_x = (cols * STRIDE_X - width)//2 + DIFF
        img = self._padding(img, border_y, border_x, value=RGB_MEAN)
        # if self._debug:
        #     image_show(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), "img padding")
        px_cls = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        px_prob = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        count = 0
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                x = j * STRIDE_X
                y = i * STRIDE_Y
                cropped_img = crop_img(img, y, x, CROP_HEIGHT, CROP_WIDTH)
                if self._debug:
                    image_show(cropped_img, "img_crop")
                crop_px_prob = self._infer(cropped_img)
                crop_px_cls = self._threshold(crop_px_prob)

                CROP_DIFF = abs(STRIDE_Y - crop_px_cls.shape[0]) // 2
                if CROP_DIFF > 0:
                    print(CROP_DIFF)
                    print(crop_px_cls.shape)
                    print(crop_px_prob.shape)
                    crop_px_cls = crop_px_cls[CROP_DIFF:-CROP_DIFF, CROP_DIFF:-CROP_DIFF]
                    crop_px_prob = crop_px_prob[CROP_DIFF:-CROP_DIFF, CROP_DIFF:-CROP_DIFF]
                    print(crop_px_cls.shape)
                    print(crop_px_prob.shape)
                if self._debug:
                    image_show(self._transform_px_cls(crop_px_cls), "px_cls")
                    image_show(self._transform_px_prob(crop_px_prob), "px_prob")
                print(np.max(crop_px_cls))
                px_cls[y+DIFF:(y+CROP_HEIGHT-DIFF),x+DIFF:(x+CROP_WIDTH-DIFF)] = crop_px_cls
                px_prob[y+DIFF:(y+CROP_HEIGHT-DIFF),x+DIFF:(x+CROP_WIDTH-DIFF)] = crop_px_prob
                count += 1

        px_prob = self._crop_border(px_prob, border_y, border_x)
        px_prob = self._transform_px_prob(px_prob)

        px_cls = self._crop_border(px_cls, border_y, border_x)
        px_cls = self._transform_px_cls(px_cls)
        return px_cls, px_prob

    def _crop_border(self, img, border_y: int, border_x: int):
        return img[border_y:-border_y, border_x:-border_x]

    def _transform_px_cls(self, px_cls):
        # px_cls = px_cls / np.max(px_cls)
        print(np.max(px_cls))
        px_cls = np.clip(px_cls, a_min=0, a_max=1)
        px_cls = np.where(px_cls > 0.5, 255, 0)
        # px_cls = cv2.applyColorMap(px_cls, cv2.COLORMAP_JET)
        return px_cls.astype(np.uint8)

    def _transform_px_prob(self, px_prob):
        px_prob = (1.0 - 2*np.abs(0.5 - px_prob))
        px_prob = np.clip(px_prob, a_min=0, a_max=1.0) * 255.0
        px_prob = np.floor(px_prob).astype(np.uint8)
        px_prob = cv2.applyColorMap(px_prob, cv2.COLORMAP_JET)
        return px_prob

    def _threshold(self, px_prob):
        return np.where(px_prob > 0.5, 1.0, 0.0)

    def _resize(self, mask, new_w, new_h):
        return grayscale_resize_nearest_uint8(mask, new_w=new_w, new_h=new_h)

    def _padding(self, img, border_y, border_x, value=0):
        return cv2.copyMakeBorder(img, border_y, border_y, border_x, border_x, cv2.BORDER_CONSTANT, value=value)


if __name__ == "__main__":
    inference = InferenceInria(debug=True, save_out=False, model_path=MODEL_1_0_6_PATH)
    inference.infer_file(SAMPLE_PATH)
