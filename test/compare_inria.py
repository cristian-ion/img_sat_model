# todo: - class-confusion matrix (use gray as color map, expect light diagonal )
# todo: - accuracy
# todo: - IoU (predict == label / )
# todo: - ROC curve - pg. 501, Computer Vision: A modern approach
# todo: - Precision as a function of recall - pg. 507, Computer Vision: A modern approach

from os import listdir
from os.path import isfile, join, basename
import cv2
import numpy as np

from inference.inference_inria import LATEST_MODEL_NAME

REPO_PATH = "/Users/cristianion/Desktop/img_sat_model"
DIR_VAL_GT = f"{REPO_PATH}/inria/AerialImageDataset/val/gt"
DIR_VAL_SEGM = f"{REPO_PATH}/inria_out/{LATEST_MODEL_NAME}/inria_val_out"
EXT_PNG = ".png"
EXT_TIF = ".tif"

POS_VALUE = 255
NEG_VALUE = 0


def create_filelist(dir, filter_ext):
    filelist = [join(dir, file) for file in listdir(dir)]
    filelist = [file for file in filelist if isfile(file)]
    filelist = [file for file in filelist if file.endswith(filter_ext)]
    filelist = [(file, file.split('.')[0]) for file in filelist]
    filelist = sorted(filelist, key=lambda x: x[1])
    filelist = [file for file, _ in filelist]
    return filelist


class CompareInria:
    def __init__(self) -> None:
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0

    def compare_dir(self, filelist_segm, filelist_gt):
        assert len(filelist_gt) == len(filelist_segm)
        print("ind", "gt_file", "segm_file", *self.metric_names, sep='\t')
        for ind, (gt_file, segm_file) in enumerate(zip(filelist_gt, filelist_segm)):
            self.compare_files(gt_file, segm_file)
            gt_filename = basename(gt_file)
            segm_filename = basename(segm_file)
            print(ind,  gt_filename, segm_filename, *self.metric_row, sep='\t')

    def compare_files(self, gt_file: str, segm_file: str):
        gt = cv2.imread(gt_file, flags=cv2.IMREAD_GRAYSCALE)
        segm = cv2.imread(segm_file, flags=cv2.IMREAD_GRAYSCALE)
        # cv2.imshow(f"gt", gt)
        # cv2.imshow(f"segm", segm)
        # cv2.waitKey()
        self.compare_gt_segm(gt, segm)

    def compare_gt_segm(self, gt, segm):
        assert gt.shape == segm.shape

        self._tp += np.sum(np.logical_and(gt == POS_VALUE, segm == POS_VALUE))
        self._fp += np.sum(np.logical_and(gt == NEG_VALUE, segm == POS_VALUE))
        self._tn += np.sum(np.logical_and(gt == NEG_VALUE, segm == NEG_VALUE))
        self._fn += np.sum(np.logical_and(gt == POS_VALUE, segm == NEG_VALUE))

    @property
    def metric_names(self):
        return ("iou", "dice", "accuracy", "error_rate")

    @property
    def metric_row(self):
        return (self.iou, self.dice, self.accuracy, self.error_rate)

    @property
    def accuracy(self):
        return (self._tp + self._tn) / (self._tp + self._fn + self._fp + self._tn)

    @property
    def error_rate(self):
        return 1 - self.accuracy

    @property
    def iou(self):
        return self._tp / (self._tp + self._fn + self._fp)

    @property
    def dice(self):
        return (2 * self._tp) / (2 * self._tp + self._fn + self._fp)

    @property
    def tpr(self):
        return self._tp / (self._tp + self._fn)

    @property
    def fpr(self):
        return self._fp / (self._fp + self._tn)

    @property
    def sensitivity(self):
        return self.tpr

    @property
    def specificity(self):
        return 1 - self.fpr


if __name__ == "__main__":
    compare = CompareInria()
    fl_out = create_filelist(DIR_VAL_SEGM, EXT_PNG)
    fl_gt = create_filelist(DIR_VAL_GT, EXT_TIF)
    compare.compare_dir(fl_out, fl_gt)

    # print(compare.metric_names)
    # print(compare.metric_row)
    # results = {name: val for name, val in zip(compare.metric_names, compare.metric_row)}
    # print(results)
    # print(LATEST_MODEL_NAME)