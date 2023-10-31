import cv2
import os


CLASSES = ["building"]
NUM_CLASSES = len(CLASSES)
CROP_HEIGHT = 512
CROP_WIDTH = 512
IMG_EXT = "tif"
GT_EXT = "tif"

ROOT_PATH = "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset"

IN_TRAIN_IMG = (
    "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/train/images"
)
IN_TRAIN_GT = (
    "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/train/gt"
)

IN_VAL_IMG = (
    "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/val/images"
)
IN_VAL_GT = (
    "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/val/gt"
)


TRAIN_IMAGES_OUT = "./_inria_train_images"
OUT_TRAIN_IMG = "./_inria_train_images/train/img"
OUT_TRAIN_GT = "./_inria_train_images/train/gt"
OUT_VAL_IMG = "./_inria_train_images/val/img"
OUT_VAL_GT = "./_inria_train_images/val/gt"

OUT_DIRS = [
    TRAIN_IMAGES_OUT,
    "./_inria_train_images/train/",
    "./_inria_train_images/val/",
    OUT_TRAIN_IMG,
    OUT_TRAIN_GT,
    OUT_VAL_IMG,
    OUT_VAL_GT,
]


def make_out_folders():
    for dir in OUT_DIRS:
        if not os.path.isdir(dir):
            os.mkdir(dir)


def padding(img):
    return cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=0)


def crop(img, y, x, h, w):
    return img[y:(y+h), x:(x+w)]


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
        for img, gt in zip(self.images, self.gts):
            name = img.split('.')[0]
            img = os.path.join(self.img_dir, img)
            gt = os.path.join(self.gt_dir, gt)
            img, gt = self.read_img_gt(img, gt)
            self.process_img_gt(img, gt, name)

    def process_img_gt(self, img, gt, name):
        img = padding(img)
        gt = padding(gt)

        h, w, _ = img.shape

        count = 0
        for y in range(0, h, CROP_HEIGHT):
            for x in range(0, w, CROP_WIDTH):
                crop_img = crop(img, y, x, CROP_HEIGHT, CROP_WIDTH)
                crop_gt = crop(gt, y, x, CROP_HEIGHT, CROP_WIDTH)
                self.save_img_gt(crop_img, crop_gt, f"{name}_{count:02d}")
                count += 1

    def read_img_gt(self, img_path, gt_path):
        print(img_path)
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        print(img.shape)
        print(gt.shape)
        return img, gt

    def save_img_gt(self, img, gt, name):
        # cv2.imshow("img", img)
        # cv2.imshow("gt", gt)
        # cv2.waitKey()
        cv2.imwrite(f"{self.out_img}/{name}.png", img)
        cv2.imwrite(f"{self.out_gt}/{name}.png", gt)


if __name__ == "__main__":
    crops_train = CropsInria(IN_TRAIN_IMG, IN_TRAIN_GT, OUT_TRAIN_IMG, OUT_TRAIN_GT)
    crops_train.process()

    crops_val = CropsInria(IN_VAL_IMG, IN_VAL_GT, OUT_VAL_IMG, OUT_VAL_GT)
    crops_val.process()
