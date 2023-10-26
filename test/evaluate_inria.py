# todo: - class-confusion matrix (use gray as color map, expect light diagonal )
# todo: - accuracy
# todo: - IoU (predict == label / )
# todo: - ROC curve - pg. 501, Computer Vision: A modern approach
# todo: - Precision as a function of recall - pg. 507, Computer Vision: A modern approach

from inference.inference_inria import InferenceInria
from os import listdir
from os.path import isfile, join


DIR_VAL_OUT = "/Users/cristianion/Desktop/img_sat_model/inria_val_out"
DIR_TEST_OUT = "/Users/cristianion/Desktop/img_sat_model/inria_test_out"

DIR_VAL = "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/val/images"
DIR_TEST = "/Users/cristianion/Desktop/img_sat_model/inria/AerialImageDataset/test/images"


def create_filelist(dir):
    filelist = [join(dir, f) for f in listdir(dir)]
    filelist = [f for f in filelist if isfile(f)]
    filelist = [f for f in filelist if f.endswith(".tif")]
    return filelist


class EvaluateInria:
    def __init__(self, dir_out) -> None:
        assert dir_out is not None
        self.infer = InferenceInria(debug=False, save_out=True, dir_out=dir_out)

    def evaluate(self, filelist):
        for file in filelist:
            self.infer.image_segment_file(file)


if __name__ == "__main__":
    eval = EvaluateInria(dir_out=DIR_VAL_OUT)
    filelist = create_filelist(DIR_VAL)
    eval.evaluate(filelist)
    print("Done val.")

    eval = EvaluateInria(dir_out=DIR_TEST_OUT)
    filelist = create_filelist(DIR_TEST)
    eval.evaluate(filelist)
    print("Done test.")
