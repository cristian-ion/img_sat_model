# todo: - class-confusion matrix (use gray as color map, expect light diagonal )
# todo: - accuracy
# todo: - IoU (predict == label / )
# todo: - ROC curve - pg. 501, Computer Vision: A modern approach
# todo: - Precision as a function of recall - pg. 507, Computer Vision: A modern approach

from os import listdir
from os.path import isfile, join
import cv2


DIR_VAL_SEGM = "/Users/cristianion/Desktop/img_sat_model/inria_val_out"
DIR_VAL_GT = "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/val/gt"
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
        print(len(filelist_gt), len(filelist_segm))
        assert len(filelist_gt) == len(filelist_segm)

        for gt_file, segm_file in zip(filelist_gt, filelist_segm):
            self.compare_files(gt_file, segm_file)

    def compare_files(self, gt_file: str, segm_file: str):
        gt = cv2.imread(gt_file, flags=cv2.IMREAD_GRAYSCALE)
        segm = cv2.imread(segm_file, flags=cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("gt", gt)
        # cv2.imshow("segm", segm)
        # cv2.waitKey()
        self.compare_gt_segm(gt, segm)

    def compare_gt_segm(self, gt, segm):
        assert gt.shape == segm.shape
        h, w = gt.shape

        for y in range(h):
            for x in range(w):
                if (gt[y,x] == POS_VALUE) and (segm[y,x] == POS_VALUE):
                    self._tp += 1
                elif (gt[y,x] == NEG_VALUE) and (segm[y,x] == POS_VALUE):
                    self._fp += 1
                elif (gt[y,x] == NEG_VALUE) and (segm[y,x] == NEG_VALUE):
                    self._tn += 1
                elif (gt[y,x] == POS_VALUE) and (segm[y,x] == NEG_VALUE):
                    self._fn += 1

        print(self.accuracy)
        print(self.error_rate)
        print(self.iou)

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


if __name__ == "__main__":
    compare = CompareInria()
    fl_out = create_filelist(DIR_VAL_SEGM, EXT_PNG)
    fl_gt = create_filelist(DIR_VAL_GT, EXT_TIF)
    compare.compare_dir(fl_out, fl_gt)

    print(compare.accuracy)
    print(compare.error_rate)
    print(compare.iou)
    print(compare.dice)
