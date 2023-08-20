import os

DSTL_ROOT_PATH = "/Users/cristianion/Desktop/satimg_data/DSTL"
DEBUG_PATH = "imgdetection/debug"

TRAIN_WKT_FILE = os.path.join(DSTL_ROOT_PATH, "train_wkt_v4.csv")
GRID_SIZES_FILE = os.path.join(DSTL_ROOT_PATH, "grid_sizes.csv")

SIXTEEN_BAND = os.path.join(DSTL_ROOT_PATH, "sixteen_band")
THREE_BAND = os.path.join(DSTL_ROOT_PATH, "three_band")

# Column names
COL_MULTIPOLYGONWKT = "MultipolygonWKT"
COL_CLASSTYPE = "ClassType"
COL_IMAGEID = "ImageId"
COL_XMAX = "Xmax"
COL_YMIN = "Ymin"

# Image extension
EXT_TIFF = ".tif"
