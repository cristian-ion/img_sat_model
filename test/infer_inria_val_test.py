# todo: - class-confusion matrix (use gray as color map, expect light diagonal )
# todo: - accuracy
# todo: - IoU (predict == label / )
# todo: - ROC curve - pg. 501, Computer Vision: A modern approach
# todo: - Precision as a function of recall - pg. 507, Computer Vision: A modern approach

from inference.inference_inria import InferenceInria, LATEST_MODEL_NAME
from os import listdir
from os.path import isfile, join


REPO_PATH = f"/Users/cristianion/Desktop/img_sat_model"
DIR_VAL_OUT = f"{REPO_PATH}/inria_out/{LATEST_MODEL_NAME}/inria_val_out"
DIR_TEST_OUT = f"{REPO_PATH}/inria_out/{LATEST_MODEL_NAME}/inria_test_out"
DIR_VAL_IN = f"{REPO_PATH}/inria/AerialImageDataset/val/images"
DIR_TEST_IN = f"{REPO_PATH}/inria/AerialImageDataset/test/images"


def create_filelist(dir):
    filelist = [join(dir, f) for f in listdir(dir)]
    filelist = [f for f in filelist if isfile(f)]
    filelist = [f for f in filelist if f.endswith(".tif")]
    return filelist


def inference_inria_val():
    filelist = create_filelist(DIR_VAL_IN)
    infer = InferenceInria(model_name=LATEST_MODEL_NAME, debug=False, save_out=True, dir_out=DIR_VAL_OUT)
    infer.infer_filelist(filelist)
    print("Done val.")


def inference_inria_test():
    filelist = create_filelist(DIR_TEST_IN)
    infer = InferenceInria(model_name=LATEST_MODEL_NAME, debug=False, save_out=True, dir_out=DIR_TEST_OUT)
    infer.infer_filelist(filelist)
    print("Done test.")


if __name__ == "__main__":
    inference_inria_val()
    inference_inria_test()
