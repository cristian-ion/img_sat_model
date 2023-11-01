import cv2
import os
import numpy as np


CLASSES = ["building"]
NUM_CLASSES = len(CLASSES)

CROP_HEIGHT = 512
CROP_WIDTH = 512

IMG_EXT = "png"
GT_EXT = "png"

OUT_TRAIN_IMG = "./_inria_train_images/train/img"
OUT_TRAIN_GT = "./_inria_train_images/train/gt"
OUT_VAL_IMG = "./_inria_train_images/val/img"
OUT_VAL_GT = "./_inria_train_images/val/gt"


class CleanCropsInria:
    def __init__(self, out_img, out_gt) -> None:
        self.images = os.listdir(out_img)
        self.gts = os.listdir(out_gt)
        self.img_dir = out_img
        self.gt_dir = out_gt

        self.images = [f for f in self.images if f.split(".")[-1] == IMG_EXT]
        self.gts = [f for f in self.gts if f.split(".")[-1] == GT_EXT]
        self.out_img = out_img
        self.out_gt = out_gt

    def process(self):
        removed = 0
        for img, gt in zip(self.images, self.gts):
            img_path = os.path.join(self.img_dir, img)
            gt_path = os.path.join(self.gt_dir, gt)
            img, gt = self.read_img_gt(img_path, gt_path)
            is_positive = self.is_positive_sample(gt)
            if not is_positive:
                removed += 1
                os.remove(img_path)
                os.remove(gt_path)
                print(removed)

    def is_positive_sample(self, gt):
        count = 0
        if np.any(gt > 0):
            count += 1
        if count == 0:
            return False
        return True

    def read_img_gt(self, img_path, gt_path):
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        return img, gt

    def save_img_gt(self, img, gt, name):
        # cv2.imshow("img", img)
        # cv2.imshow("gt", gt)
        # cv2.waitKey()
        cv2.imwrite(f"{self.out_img}/{name}.png", img)
        cv2.imwrite(f"{self.out_gt}/{name}.png", gt)


if __name__ == "__main__":
    crops_train = CleanCropsInria(OUT_TRAIN_IMG, OUT_TRAIN_GT)
    crops_train.process()

    crops_val = CleanCropsInria(OUT_VAL_IMG, OUT_VAL_GT)
    crops_val.process()
