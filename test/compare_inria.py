# todo: - class-confusion matrix (use gray as color map, expect light diagonal )
# todo: - accuracy
# todo: - IoU (predict == label / )
# todo: - ROC curve - pg. 501, Computer Vision: A modern approach
# todo: - Precision as a function of recall - pg. 507, Computer Vision: A modern approach

from os import listdir
from os.path import isfile, join


DIR_VAL_OUT = "/Users/cristianion/Desktop/img_sat_model/inria_val_out"
DIR_VAL_GT = "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/val/gt"
EXT_PNG = ".png"
EXT_TIF = ".tif"

def create_filelist(dir, filter_ext):
    filelist = [join(dir, f) for f in listdir(dir)]
    filelist = [f for f in filelist if isfile(f)]
    filelist = [f for f in filelist if f.endswith(filter_ext)]
    return filelist


class CompareInria:
    def __init__(self) -> None:
        pass

    def compare_dir(self, filelist_out, filelist_gt):
        print(len(filelist_gt), len(filelist_out))
        assert len(filelist_gt) == len(filelist_out)


if __name__ == "__main__":
    eval = CompareInria()
    fl_out = create_filelist(DIR_VAL_OUT, EXT_PNG)
    fl_gt = create_filelist(DIR_VAL_GT, EXT_TIF)
    eval.compare_dir(fl_out, fl_gt)
