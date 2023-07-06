import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
import contextlib
import random
import platform
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
from torch import nn


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

print(platform.platform())  # print current platform
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


RESISC45_DIRPATH = "/Users/cristianion/Desktop/satimg_data/NWPU-RESISC45"

RESISC45_LABELS = [
    'forest',
    'railway_station',
    'tennis_court',
    'basketball_court',
    'river',
    'storage_tank',
    'harbor',
    'terrace',
    'thermal_power_station',
    'golf_course',
    'runway',
    'roundabout',
    'bridge',
    'industrial_area',
    'baseball_diamond',
    'mobile_home_park',
    'overpass',
    'church',
    'chaparral',
    'railway',
    'stadium',
    'medium_residential',
    'sea_ice',
    'intersection',
    'lake',
    'palace',
    'airplane',
    'cloud',
    'sparse_residential',
    'airport',
    'snowberg',
    'parking_lot',
    'commercial_area',
    'rectangular_farmland',
    'island',
    'beach',
    'circular_farmland',
    'dense_residential',
    'ship',
    'mountain',
    'desert',
    'freeway',
    'meadow',
    'wetland',
    'ground_track_field',
]

UCMERCED_LANDUSE_DIRPATH = "/Users/cristianion/Desktop/satimg_data/UCMerced_LandUse/Images"
UCMERCED_LANDUSE_LABELS = [
    "forest",
    "buildings",
    "river",
    "mobilehomepark",
    "harbor",
    "golfcourse",
    "agricultural",
    "runway",
    "baseballdiamond",
    "overpass",
    "chaparral",
    "tenniscourt",
    "intersection",
    "airplane",
    "parkinglot",
    "sparseresidential",
    "mediumresidential",
    "denseresidential",
    "beach",
    "freeway",
    "storagetanks",
]

EUROSAT_DIRPATH = "/Users/cristianion/Desktop/satimg_data/EuroSAT_RGB"
EUROSAT_LABELS = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


RESISC45_DATASET_FILE = "resisc45_dataset.csv"
UCMERCEDLU_DATASET_FILE = "ucmercedlu_dataset.csv"
EUROSAT_DATASET_FILE = "eurosat_dataset.csv"

K_FOLDS = 10


def _partition_label_in_folds(data, label, num_folds):
    data_by_label = data[data.label == label]

    print(f"Number of samples {len(data_by_label)} {label}")

    out, bins = pd.cut(np.arange(len(data_by_label)), bins=num_folds, include_lowest=False, labels=[i for i in range(num_folds)], retbins=True)
    # print(bins.tolist())
    fold = pd.Series(out)

    print(f"Folds: \n{fold.value_counts()}")

    print(len(fold))

    fold_index = list(data_by_label.columns).index('fold')
    index_fold = zip(data_by_label.index, fold)

    for i, f in index_fold:
        data.iloc[i, fold_index] = f


def _parition_dataset_in_folds(data, num_folds):
    # partition the dataset into folds
    parition_size = int(100 / num_folds)
    print(f"Partition size {parition_size}")

    data = data.sample(frac=1).reset_index(drop=True)  # resample dataset randomly.

    labels = data['label'].unique()

    data['fold'] = np.zeros(len(data), dtype=int)

    for label in labels:
        _partition_label_in_folds(data, label, num_folds)

    data.sort_values(by=['fold'], inplace=True)  # sort values
    return data


def dataset_imgpath_label(out_file: str, input_dir, labels, num_folds=K_FOLDS):
    filelist = [(f"{input_dir}/{f}", labels[labels.index(f)]) for f in os.listdir(input_dir) if f in labels]
    df = pd.DataFrame(filelist, columns=['dirpath', 'label'])
    print(df.head())
    print(df.info())


    f = open(out_file + ".stats", "w")


    data = []
    for i, input_dir in enumerate(df["dirpath"]):
        images = os.listdir(input_dir)
        images = [f"{input_dir}/{img}" for img in images]
        rows = [(img, df['label'][i]) for img in images]
        data.extend(rows)
    data = pd.DataFrame(data, columns=["imgpath", "label"])

    label_index = []
    for t in data.label:
        label_index.append(labels.index(t))
    data['label_index'] = label_index

    f.write(str(data.head()))
    f.write(str(data.info()))
    f.write(str(data['label'].value_counts()))
    f.write(str(data['label_index'].value_counts()))

    # partition the dataset in K-Folds
    data = _parition_dataset_in_folds(data, num_folds=num_folds)

    for fold in range(num_folds):
        f.write(f"Fold {fold} stats:\n")
        f.write(str(data[data['fold'] == fold]['label'].value_counts()))

    # save dataset on disk
    data.to_csv(out_file, index=False)

    f.close()

    return data


def adapter_resisc():
    pd.DataFrame({'label': RESISC45_LABELS}).to_csv("resisc45_labels.csv", index=False, header=False)
    if os.path.exists(RESISC45_DATASET_FILE):
        print(f"{RESISC45_DATASET_FILE} already exists.")
        return
    print(len(RESISC45_LABELS))  # number of expected labels
    resisc45_data = dataset_imgpath_label(RESISC45_DATASET_FILE, RESISC45_DIRPATH, RESISC45_LABELS)
    print(len(resisc45_data))


def adapter_ucmerced():
    pd.DataFrame({'label': UCMERCED_LANDUSE_LABELS}).to_csv("ucmercedlu_labels.csv", index=False, header=False)
    if os.path.exists(UCMERCEDLU_DATASET_FILE):
        print(f"{UCMERCEDLU_DATASET_FILE} already exists.")
        return
    print(len(UCMERCED_LANDUSE_LABELS))
    ucmercedlu_data = dataset_imgpath_label(UCMERCEDLU_DATASET_FILE, UCMERCED_LANDUSE_DIRPATH, UCMERCED_LANDUSE_LABELS)
    print(len(ucmercedlu_data))


def adapter_eurosat():
    pd.DataFrame({'label': EUROSAT_LABELS}).to_csv("eurosat_labels.csv", index=False, header=False)
    if os.path.exists(EUROSAT_DATASET_FILE):
        print(f"{EUROSAT_DATASET_FILE} already exists.")
        return
    print(len(EUROSAT_LABELS))
    eurosat_data = dataset_imgpath_label(EUROSAT_DATASET_FILE, EUROSAT_DIRPATH, EUROSAT_LABELS)
    print(len(eurosat_data))


if __name__ == "__main__":
    adapter_resisc()
    adapter_ucmerced()
    adapter_eurosat()
    print("done.")
