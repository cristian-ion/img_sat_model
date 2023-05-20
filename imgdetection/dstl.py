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

from dstl_config import *
from unet import UNet



def convert_coordinates_to_raster(coords, xmax, ymin, width, height):
    w1 = 1.0 * width * width / (width + 1)
    h1 = 1.0 * height * height / (height + 1)
    xf = w1 / xmax
    yf = h1 / ymin
    coords[:, 0] *= xf
    coords[:, 1] *= yf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


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
    imgpath = os.path.join(TIFFDIR_B3, imageid + EXT_TIFF)
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


def load_dstl_dataset():
    df_wkt = pd.read_csv(TRAINSET_FILE)
    df_gs = pd.read_csv(GRIDSIZES_FILE)

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



class DstlDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        df_wkt = pd.read_csv(TRAINSET_FILE)
        df_gs = pd.read_csv(GRIDSIZES_FILE)
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


if __name__ == "__main__":
    # computer some stats
    # load_dstl_dataset()

    # train
    ds = DstlDataset()
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    model = UNet(3, 2, False)
    device = get_device()
    model.to(device)

    model.train()  # set model to train mode

    train_loss = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for batch, (X, y) in enumerate(dl):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
