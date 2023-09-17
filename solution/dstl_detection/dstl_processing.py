import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import tifffile
from shapely.wkt import loads as wkt_loads

from .dstl_constants import (EXT_TIFF, GRID_SIZES_FILE, THREE_BAND,
                             TRAIN_WKT_FILE)


def read_grid_sizes(grid_sizes_csv=GRID_SIZES_FILE):
    return pd.read_csv(grid_sizes_csv, names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


def read_train_wkt(train_wkt_csv=TRAIN_WKT_FILE):
    return pd.read_csv(train_wkt_csv)


class DstlProcessing:
    def __init__(self, df_train_wkt, df_grid_sizes, classes) -> None:
        self.df_train_wkt = df_train_wkt
        self.df_grid_sizes =  df_grid_sizes
        self.classes = classes

    def read_image_and_mask(self, raster_size, image_id):
        img_path = os.path.join(THREE_BAND, image_id + EXT_TIFF)
        img = tifffile.imread(img_path)

        if img.shape[0] == 3:
            img = np.rollaxis(img, 0, 3)

        masks = [self.generate_mask_for_image_and_class(raster_size, image_id, i) for i in range(len(self.classes))]

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
        xmax, ymin = self.df_grid_sizes[self.df_grid_sizes.ImageId == imageId].iloc[0, 1:].astype(float)
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
            raster_coords = DstlProcessing._convert_coordinates_to_raster(vector_coords, raster_img_size, xymax)
            raster_coords_list.append(raster_coords)
            for pi in poly.interiors:
                interior = np.array(list(pi.coords))
                interior_c = DstlProcessing._convert_coordinates_to_raster(interior, raster_img_size, xymax)
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


if __name__ == "__main__":
    read_train_wkt()
    read_grid_sizes()