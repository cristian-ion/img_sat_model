import cv2
import os
import math

from constants import REPO_DIR


CLASSES = ["building"]
NUM_CLASSES = len(CLASSES)
IMG_CROP_HEIGHT = 572
IMG_CROP_WIDTH = 572
GT_CROP_HEIGHT = 388
GT_CROP_WIDTH = 388
IMG_EXT = "tif"
GT_EXT = "tif"

INRIA_PATH = f"{REPO_DIR}/inria/AerialImageDataset"
IN_TRAIN_IMG = f"{INRIA_PATH}/train/images"
IN_TRAIN_GT = f"{INRIA_PATH}/train/gt"
IN_VAL_IMG = f"{INRIA_PATH}/val/images"
IN_VAL_GT = f"{INRIA_PATH}/val/gt"

TRAIN_IMAGES_OUT = "./_inria_train_images"
OUT_TRAIN_IMG = f"./_inria_train_images/{IMG_CROP_HEIGHT}_{GT_CROP_HEIGHT}/train/img"
OUT_TRAIN_GT = f"./_inria_train_images/{IMG_CROP_HEIGHT}_{GT_CROP_HEIGHT}/train/gt"
OUT_VAL_IMG = f"./_inria_train_images/{IMG_CROP_HEIGHT}_{GT_CROP_HEIGHT}/val/img"
OUT_VAL_GT = f"./_inria_train_images/{IMG_CROP_HEIGHT}_{GT_CROP_HEIGHT}/val/gt"

OUT_DIRS = [
    TRAIN_IMAGES_OUT,
    OUT_TRAIN_IMG,
    OUT_TRAIN_GT,
    OUT_VAL_IMG,
    OUT_VAL_GT,
]


def make_out_folders():
    for dir in OUT_DIRS:
        if not os.path.isdir(dir):
            os.makedirs(dir)


def padding(img, border_size, value=0):
    return cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=value)


def crop(img, y, x, h, w):
    assert y + h <= img.shape[0]
    assert x + w <= img.shape[1], f"{x}, {x+w}, {img.shape[1]}"
    img = img[y:(y+h), x:(x+w)]
    assert img.shape[0] == h
    assert img.shape[1] == w
    return img


class CropsInria:
    def __init__(self, images_dir, gt_dir, out_img, out_gt) -> None:
        make_out_folders()
        self.images = os.listdir(images_dir)
        self.gts = os.listdir(gt_dir)
        self.img_dir = images_dir
        self.gt_dir = gt_dir

        self.images = [f for f in self.images if f.split(".")[-1] == IMG_EXT]
        self.gts = [f for f in self.gts if f.split(".")[-1] == GT_EXT]
        self.out_img = out_img
        self.out_gt = out_gt

    def process(self):
        count = 0
        for img, gt in zip(self.images, self.gts):
            name = img.split('.')[0]
            img = os.path.join(self.img_dir, img)
            gt = os.path.join(self.gt_dir, gt)
            img, gt = self.read_img_gt(img, gt)
            self.process_img_gt(img, gt, name)
            count += 1

    def process_img_gt(self, img, gt, name):
        img_size = (img.shape[0], img.shape[1])
        gt_size = (gt.shape[0], gt.shape[1])
        assert img_size == gt_size

        height = img.shape[0]
        gt_border_size = abs(height - (math.ceil(height / GT_CROP_HEIGHT) * GT_CROP_HEIGHT))//2
        img_border_size = gt_border_size + (IMG_CROP_HEIGHT - GT_CROP_HEIGHT)//2

        gt = padding(gt, border_size=gt_border_size)
        img = padding(img, border_size=img_border_size)

        h, w, _ = gt.shape

        count = 0
        for y in range(0, h, GT_CROP_HEIGHT):
            for x in range(0, w, GT_CROP_WIDTH):
                crop_gt = crop(gt, y, x, GT_CROP_HEIGHT, GT_CROP_WIDTH)
                crop_img = crop(img, y, x, IMG_CROP_HEIGHT, IMG_CROP_WIDTH)
                self.save_img_gt(crop_img, crop_gt, f"{name}_{count:02d}")
                count += 1

    def read_img_gt(self, img_path, gt_path):
        print(img_path)
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        return img, gt

    def save_img_gt(self, img, gt, name):
        cv2.imwrite(f"{self.out_img}/{name}.png", img)
        cv2.imwrite(f"{self.out_gt}/{name}.png", gt)

if __name__ == "__main__":
    crops_train = CropsInria(IN_TRAIN_IMG, IN_TRAIN_GT, OUT_TRAIN_IMG, OUT_TRAIN_GT)
    crops_train.process()

    crops_val = CropsInria(IN_VAL_IMG, IN_VAL_GT, OUT_VAL_IMG, OUT_VAL_GT)
    crops_val.process()
