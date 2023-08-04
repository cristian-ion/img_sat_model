import os

DSTL_ROOT_PATH = "/Users/cristianion/Desktop/satimg_data/DSTL"
DEBUG_PATH = "imgdetection/debug"

TRAINSET_FILE = os.path.join(DSTL_ROOT_PATH, "train_wkt_v4.csv")
GRIDSIZES_FILE = os.path.join(DSTL_ROOT_PATH, "grid_sizes.csv")

TIFFDIR_B16 = os.path.join(DSTL_ROOT_PATH, "sixteen_band")
TIFFDIR_B3 = os.path.join(DSTL_ROOT_PATH, "three_band")

# Column names
COL_MULTIPOLYGONWKT = "MultipolygonWKT"
COL_CLASSTYPE = "ClassType"
COL_IMAGEID = "ImageId"
COL_XMAX = "Xmax"
COL_YMIN = "Ymin"

# Image extension
EXT_TIFF = ".tif"

COLORS = {
    "black": (0, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
    "brown": (0, 75, 150),
    "pink": (147, 20, 255),
    "yellow": (0, 255, 255),
}