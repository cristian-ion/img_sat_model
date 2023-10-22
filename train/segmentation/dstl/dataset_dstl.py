import os
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from albumentations.pytorch import ToTensorV2
from shapely.wkt import loads as wkt_loads

MAJOR_VERSION = 1
DSTL_ROOT_PATH = "/Users/cristianion/Desktop/img_sat_data/DSTL"
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
NUM_CLASSES = len(CLASSES)

IMAGE_RES_X = 512
IMAGE_RES_Y = 512
BATCH_SIZE = 4
VAL_BATCH_SIZE = 1

DSTL_NAMECODE = "dstl"

VALIDATION_IMAGE_IDS = {"6070_2_3", "6010_1_2", "6040_4_4", "6100_2_2"}


class DstlProcessing:
    def __init__(self, df_train_wkt, df_grid_sizes, classes) -> None:
        self.df_train_wkt = df_train_wkt
        self.df_grid_sizes = df_grid_sizes
        self.classes = classes

    def read_image_and_mask(self, raster_size, image_id):
        img_path = os.path.join(THREE_BAND, image_id + EXT_TIFF)
        img = tifffile.imread(img_path)

        if img.shape[0] == 3:
            img = np.rollaxis(img, 0, 3)

        masks = [
            self.generate_mask_for_image_and_class(raster_size, image_id, i)
            for i in range(len(self.classes))
        ]

        return img, np.array(masks)

    def generate_mask_for_image_and_class(self, raster_size, image_id, class_index):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        xymax = self._get_xmax_ymin(image_id)
        polygon_list = self._get_polygon_list(image_id, class_index)
        contours = self._get_and_convert_contours(polygon_list, raster_size, xymax)
        mask = self._plot_mask_from_contours(raster_size, contours, class_index)
        return mask

    def _get_xmax_ymin(self, imageId: int):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        xmax, ymin = (
            self.df_grid_sizes[self.df_grid_sizes.ImageId == imageId]
            .iloc[0, 1:]
            .astype(float)
        )
        return (xmax, ymin)

    @staticmethod
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

    def _get_and_convert_contours(self, multipolygon, raster_img_size, xymax):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        raster_coords_list = []
        raster_interior_list = []
        if multipolygon is None:
            return None

        for poly in multipolygon.geoms:
            vector_coords = np.array(list(poly.exterior.coords))
            raster_coords = DstlProcessing._convert_coordinates_to_raster(
                vector_coords, raster_img_size, xymax
            )
            raster_coords_list.append(raster_coords)
            for pi in poly.interiors:
                interior = np.array(list(pi.coords))
                interior_c = DstlProcessing._convert_coordinates_to_raster(
                    interior, raster_img_size, xymax
                )
                raster_interior_list.append(interior_c)
        return raster_coords_list, raster_interior_list

    def _plot_mask_from_contours(self, raster_size, contours, class_index):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        # binary mask.
        img_mask = np.zeros(raster_size, np.uint8)
        if contours is None:
            return img_mask
        perim_list, interior_list = contours
        cv2.fillPoly(img_mask, perim_list, 255)
        cv2.fillPoly(img_mask, interior_list, 0)
        # cv2.imwrite(f"image_{self.classes[class_index]}_{datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S')}.png", img_mask)
        return img_mask

    def _get_polygon_list(self, image_id, class_type):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        df_image = self.df_train_wkt[self.df_train_wkt.ImageId == image_id]
        multipoly_def = df_image[df_image.ClassType == class_type].MultipolygonWKT
        polygons = None
        if len(multipoly_def) > 0:
            assert len(multipoly_def) == 1
            polygons = wkt_loads(multipoly_def.values[0])
        return polygons


class DstlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        transform,
        train_csv,
        grid_csv,
        classes,
        train_res_y,
        train_res_x,
        image_ids: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        train_df = pd.read_csv(train_csv)
        grid_df = pd.read_csv(
            grid_csv, names=[IMAGEID_COLUMN, XMAX_COLUMN, YMIN_COLUMN], skiprows=1
        )

        if image_ids:
            train_df = train_df[train_df[IMAGEID_COLUMN].isin(image_ids)]

        self.train_df = train_df
        self.train_res_y = train_res_y
        self.train_res_x = train_res_x
        self.transform = transform
        self.processing = DstlProcessing(train_df, grid_df, classes)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, index):
        sample = self.train_df.iloc[index]

        image, mask = self.processing.read_image_and_mask(
            raster_size=(self.train_res_y, self.train_res_x),
            image_id=sample[IMAGEID_COLUMN],
        )

        mask[mask == 255] = 1
        mask = mask.astype(np.float32)

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        mask = torch.from_numpy(mask)
        return image, mask


class DstlTrainConfig:
    def __init__(self) -> None:
        DSTL_TRAIN_TRANSFORM = A.Compose(
            [
                A.Resize(height=IMAGE_RES_Y, width=IMAGE_RES_X),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        DSTL_VAL_TRANSFORM = A.Compose(
            [
                A.Resize(height=IMAGE_RES_Y, width=IMAGE_RES_X),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        self._trainset = DstlDataset(
            transform=DSTL_TRAIN_TRANSFORM,
            train_csv=TRAIN_WKT_FILE,
            grid_csv=GRID_SIZES_FILE,
            classes=CLASSES,
            train_res_x=IMAGE_RES_X,
            train_res_y=IMAGE_RES_Y,
            image_ids=None,
        )
        self._valset = DstlDataset(
            transform=DSTL_VAL_TRANSFORM,
            train_csv=TRAIN_WKT_FILE,
            grid_csv=GRID_SIZES_FILE,
            classes=CLASSES,
            train_res_x=IMAGE_RES_X,
            train_res_y=IMAGE_RES_Y,
            image_ids=VALIDATION_IMAGE_IDS,
        )

        self._criterion = torch.nn.CrossEntropyLoss()

    @property
    def trainset(self):
        return self._trainset

    @property
    def valset(self):
        return self._valset

    @property
    def num_classes(self):
        return NUM_CLASSES

    @property
    def batch_size(self):
        return BATCH_SIZE

    @property
    def val_batch_size(self):
        return 1

    @property
    def namecode(self):
        return DSTL_NAMECODE

    @property
    def criterion(self):
        return self._criterion

    @property
    def major_version(self):
        return MAJOR_VERSION
