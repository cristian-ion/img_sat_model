import os

DSTL_ROOT_PATH = "/Users/cristianion/Desktop/satimg_data/DSTL"
DEBUG_PATH = "imgdetection/debug"

TRAIN_WKT_FILE = os.path.join(DSTL_ROOT_PATH, "train_wkt_v4.csv")
GRID_SIZES_FILE = os.path.join(DSTL_ROOT_PATH, "grid_sizes.csv")

SIXTEEN_BAND = os.path.join(DSTL_ROOT_PATH, "sixteen_band")
THREE_BAND = os.path.join(DSTL_ROOT_PATH, "three_band")

# Column names
MPWKT_COLUMN = "MultipolygonWKT"
CLASSTYPE_COLUMN = "ClassType"
IMAGEID_COLUMN = "ImageId"
XMAX_COLUMN = "Xmax"
YMIN_COLUMN = "Ymin"

# Image extension
EXT_TIFF = ".tif"

CLASSES = [
    "building",
    "structures",
    "road",
    "track",
    "tree",
    "crops",
    "waterway",
    "standing_water",
    "vehicle_large",
    "vehicle_small",
]

IMAGE_RES_X = 512
IMAGE_RES_Y = 512
