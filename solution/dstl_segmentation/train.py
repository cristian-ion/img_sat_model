import pandas as pd
import tifffile as tiff
import seaborn as sns
import json
import shapely
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose
from torchvision import transforms
from shapely.wkt import loads as wkt_loads

from solution.building_segmentation.unet import UNet


import os

DSTL_ROOT_PATH = "/Users/cristianion/Desktop/satimg_data/DSTL"
DEBUG_PATH = "dstl_debug"

# All file paths
TRAIN_WKT_FILE = os.path.join(DSTL_ROOT_PATH, "train_wkt_v4.csv")
GRID_SIZES_FILE = os.path.join(DSTL_ROOT_PATH, "grid_sizes.csv")
IMAGES_DIR_BANDS_16 = os.path.join(DSTL_ROOT_PATH, "sixteen_band")
IMAGES_DIR_BANDS_3 = os.path.join(DSTL_ROOT_PATH, "three_band")

# All column names
COL_MULTIPOLYGONWKT = "MultipolygonWKT"
COL_CLASSTYPE = "ClassType"
COL_IMAGEID = "ImageId"
COL_XMAX = "Xmax"
COL_YMIN = "Ymin"

# Image extension
EXT_TIFF = ".tif"

IMAGE_RES_X = 512
IMAGE_RES_Y = 512


def convert_coordinates_to_raster(coords, xmax, ymin, width, height):
    w1 = 1.0 * width * width / (width + 1)
    h1 = 1.0 * height * height / (height + 1)
    xf = w1 / xmax
    yf = h1 / ymin
    coords[:, 0] *= xf
    coords[:, 1] *= yf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int

def _get_xmax_ymin(grid_sizes_panda, imageId):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _convert_coordinates_to_raster(coords, img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask



def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def process_polylist(ob, imageid, classtype, xmax, ymin, width, height):
    # todo: save poly stats (exterior and interior lengths)
    print(len(ob.geoms))

    perim_list = []
    interior_list = []

    for i, poly in enumerate(ob.geoms):
        # print(imageid, classtype, i, poly.area, poly.length)
        # print(f"[{i}]exterior length = {poly.exterior.length}")
        coords = np.array(list(poly.exterior.coords))
        coords = convert_coordinates_to_raster(coords, xmax, ymin, width, height)
        perim_list.append(coords)

        for j, poly_interior in enumerate(poly.interiors):
            # print(f"[{i}] {imageid} interior length = {poly_interior.length}")
            coords = np.array(list(poly_interior.coords))
            coords = convert_coordinates_to_raster(coords, xmax, ymin, width, height)
            interior_list.append(coords)

    return perim_list, interior_list


def process_train_sample(imageid, classtype, mpwkt, xmax, ymin):
    imgpath = os.path.join(IMAGES_DIR_BANDS_3, imageid + EXT_TIFF)
    img = tiff.imread(imgpath)
    # plt.figure()
    # fig = tiff.imshow(img)
    # plt.savefig(os.path.join(DEBUG_PATH, f"{imageid}_{classtype}.png"))

    c, w, h = img.shape

    # print(f"width={w} height={h} channels={c}")

    ob = shapely.from_wkt(mpwkt) # A collection of one or more Polygons.
    exteriors, interiors = process_polylist(ob, imageid, classtype, xmax, ymin, w, h)

    # create image mask
    # print(imageid, classtype)
    img_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(img_mask, exteriors, color=1)
    cv2.fillPoly(img_mask, interiors, color=0)
    # plt.imshow(img_mask)
    # plt.savefig(os.path.join(DEBUG_PATH, f"{imageid}_{classtype}_mask.png"))

    return img, img_mask


def run_stats():
    df_wkt = pd.read_csv(TRAIN_WKT_FILE)
    df_gs = pd.read_csv(GRID_SIZES_FILE)

    df_gs.rename(columns={'Unnamed: 0': COL_IMAGEID}, inplace=True)

    print(df_gs.info())
    print(df_gs.head())
    print(df_gs.describe().T)

    # join wkt and gs
    df_wkt = pd.merge(left=df_wkt, right=df_gs, on=COL_IMAGEID)
    print(df_wkt.head())
    print(df_wkt.info())


    all_imgmax = [None, None, None]
    all_imgmin = [None, None, None]
    for wkt_i in range(max(5, len(df_wkt))):
        img, img_mask = process_train_sample(
            df_wkt.iloc[wkt_i][COL_IMAGEID],
            df_wkt.iloc[wkt_i][COL_CLASSTYPE],
            df_wkt.iloc[wkt_i][COL_MULTIPOLYGONWKT],
            df_wkt.iloc[wkt_i][COL_XMAX],
            df_wkt.iloc[wkt_i][COL_YMIN],
            )

        for i in range(3):
            imgmax = np.max(img[i])
            imgmin = np.min(img[i])

            if all_imgmax[i] is None or imgmax > all_imgmax[i]:
                all_imgmax[i] = imgmax
            if all_imgmin[i] is None or imgmin < all_imgmin[i]:
                all_imgmin[i] = imgmin

    stats = {
        "min_r": [all_imgmin[0]],
        "max_r": [all_imgmax[0]],

        "min_g": [all_imgmin[1]],
        "max_g": [all_imgmax[1]],

        "min_b": [all_imgmin[2]],
        "max_b": [all_imgmax[2]],
    }

    pd.DataFrame(stats).to_csv("dstl_img_stats.csv", index=False)

def crop_center(img, cropx, cropy):
    c, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty+cropy,startx:startx+cropx]


def crop_center_mask(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy,startx:startx+cropx]


class DstlDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        df_wkt = pd.read_csv(TRAIN_WKT_FILE)
        df_gs = pd.read_csv(GRID_SIZES_FILE)
        df_gs.rename(columns={'Unnamed: 0': COL_IMAGEID}, inplace=True)
        df_wkt = pd.merge(left=df_wkt, right=df_gs, on=COL_IMAGEID)

        self.df = df_wkt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        img, img_mask = process_train_sample(
            sample[COL_IMAGEID],
            sample[COL_CLASSTYPE],
            sample[COL_MULTIPOLYGONWKT],
            sample[COL_XMAX],
            sample[COL_YMIN],
        )
        print(img.shape)

        # min-max normalization
        # img[0, :] = (img[0, :] - 1) / (2047 - 1)
        # img[1, :] = (img[1, :] - 157) / (2047 - 157)
        # img[2, :] = (img[2, :] - 91) / (2047 - 91)

        img = img.astype(np.float32) - 1024.0

        # center crop
        img = crop_center(img, 256, 256)
        img_mask = crop_center_mask(img_mask, 256, 256)

        return torch.from_numpy(img).float(), torch.from_numpy(img_mask).float()


def get_device():
    # find CUDA / MPS / CPU device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def run_tests():
    assert True is False


def run_train():
    ds = DstlDataset()
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    size = len(ds)

    # Create UNET
    model = UNet(in_channels=3, n_classes=10, bilinear=True)
    device = get_device()
    model.to(device)

    # Set model to train mode, necessary before backprop
    model.train()

    train_loss = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for batch, (X, y) in enumerate(dl):
        X, y = X.to(device), y.to(device)

        pred = model(X)  # forward
        loss = criterion(pred, y)  # prediction error / loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    run_tests()
    run_stats()
    # run_train()
