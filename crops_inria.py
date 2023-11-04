import cv2
import os
import math

from constants import REPO_DIR

CLASSES = ["building"]
NUM_CLASSES = len(CLASSES)
CROP_HEIGHT = 572
CROP_WIDTH = 572
STRIDE_Y = 388
STRIDE_X = 388
IMG_EXT = "tif"
GT_EXT = "tif"

INRIA_PATH = f"{REPO_DIR}/inria/AerialImageDataset"
IN_TRAIN_IMG = f"{INRIA_PATH}/train/images"
IN_TRAIN_GT = f"{INRIA_PATH}/train/gt"
IN_VAL_IMG = f"{INRIA_PATH}/val/images"
IN_VAL_GT = f"{INRIA_PATH}/val/gt"

TRAIN_IMAGES_OUT = "./_inria_train_images"
OUT_TRAIN_IMG = f"./_inria_train_images/{CROP_HEIGHT}_2/train/img"
OUT_TRAIN_GT = f"./_inria_train_images/{CROP_HEIGHT}_2/train/gt"
OUT_VAL_IMG = f"./_inria_train_images/{CROP_HEIGHT}_2/val/img"
OUT_VAL_GT = f"./_inria_train_images/{CROP_HEIGHT}_2/val/gt"

OUT_DIRS = [
    TRAIN_IMAGES_OUT,
    OUT_TRAIN_IMG,
    OUT_TRAIN_GT,
    OUT_VAL_IMG,
    OUT_VAL_GT,
]

COLOR_MEAN = [math.ceil(0.485 * 255), math.ceil(0.456 * 255), math.ceil(0.406 * 255)]

def make_out_folders():
    for dir in OUT_DIRS:
        if not os.path.isdir(dir):
            os.makedirs(dir)


def padding(img, border_size:tuple, value=0):
    return cv2.copyMakeBorder(img, border_size[0], border_size[1], border_size[2], border_size[3], cv2.BORDER_CONSTANT, value=value)


def crop(img, y=None, x=None, h=None, w=None, border=None):
    y = y or 0
    x = x or 0
    h = h or img.shape[0]
    w = w or img.shape[1]
    border = border or 0
    img = img[y+border:(y+h-border), x+border:(x+w-border)]
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
        width = img.shape[1]

        rows = math.ceil(height / STRIDE_Y)
        cols = math.ceil(width / STRIDE_X)
        bd_thick_y = (rows * STRIDE_Y - height)//2 + 92
        bd_thick_x = (cols * STRIDE_X - width)//2 + 92
        border_size = (bd_thick_y, bd_thick_y, bd_thick_x, bd_thick_x)
        gt = padding(gt, border_size=border_size, value=0)
        img = padding(img, border_size=border_size, value=COLOR_MEAN)

        count = 0
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                x = j * STRIDE_X
                y = i * STRIDE_Y
                crop_gt = crop(gt, y, x, CROP_HEIGHT, CROP_WIDTH)
                crop_img = crop(img, y, x, CROP_HEIGHT, CROP_WIDTH)
                # cropped = crop(crop_gt, border=92)
                # cv2.imshow(f"gt {crop_gt.shape}", crop_gt)
                # cv2.imshow(f"crop gt {cropped.shape}", cropped)
                self.save_img_gt(crop_img, crop_gt, f"{name}_{count:02d}")
                count += 1
                # cv2.waitKey()

    def read_img_gt(self, img_path, gt_path):
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        return img, gt

    def save_img_gt(self, img, gt, name):
        cv2.imwrite(f"{self.out_img}/{name}.png", img)
        cv2.imwrite(f"{self.out_gt}/{name}.png", gt)

if __name__ == "__main__":
    crops_val = CropsInria(IN_VAL_IMG, IN_VAL_GT, OUT_VAL_IMG, OUT_VAL_GT)
    crops_val.process()

    crops_train = CropsInria(IN_TRAIN_IMG, IN_TRAIN_GT, OUT_TRAIN_IMG, OUT_TRAIN_GT)
    crops_train.process()
