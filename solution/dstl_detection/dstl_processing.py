import cv2
import numpy as np
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os

from .dstl_constants import GRID_SIZES_FILE, TRAIN_WKT_FILE, THREE_BAND, EXT_TIFF


def read_grid_sizes(grid_sizes_csv=GRID_SIZES_FILE):
    return pd.read_csv(grid_sizes_csv, names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


def read_train_wkt(train_wkt_csv=TRAIN_WKT_FILE):
    return pd.read_csv(train_wkt_csv)


class DstlProcessingLib:
    def __init__(self, df_train_wkt, df_grid_sizes) -> None:
        self.df_train_wkt = df_train_wkt
        self.df_grid_sizes =  df_grid_sizes
        self.num_classes = 10

    def read_image_and_mask(self, raster_size, image_id):
        img_path = os.path.join(THREE_BAND, image_id + EXT_TIFF)
        img = tiff.imread(img_path)

        masks = [self.generate_mask_for_image_and_class(raster_size, image_id, i) for i in range(len(self.num_classes))]

        return img, masks

    def generate_mask_for_image_and_class(self, raster_size, image_id, class_type):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        xymax = self._get_xmax_ymin(self.df_grid_sizes, image_id)
        polygon_list = self._get_polygon_list(self.df_train_wkt, image_id, class_type)
        contours = self._get_and_convert_contours(polygon_list, raster_size, xymax)
        mask = self._plot_mask_from_contours(raster_size, contours, 1)
        return mask

    def _get_xmax_ymin(self, grid_sizes_df, imageId: int):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        xmax, ymin = grid_sizes_df[grid_sizes_df.ImageId == imageId].iloc[0, 1:].astype(float)
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

    def _get_and_convert_contours(self, polygon_list, raster_img_size, xymax):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        perim_list = []
        interior_list = []
        if polygon_list is None:
            return None
        for k in range(len(polygon_list)):
            poly = polygon_list[k]
            perim = np.array(list(poly.exterior.coords))
            perim_c = self._convert_coordinates_to_raster(perim, raster_img_size, xymax)
            perim_list.append(perim_c)
            for pi in poly.interiors:
                interior = np.array(list(pi.coords))
                interior_c = self._convert_coordinates_to_raster(interior, raster_img_size, xymax)
                interior_list.append(interior_c)
        return perim_list, interior_list

    def _plot_mask_from_contours(self, raster_size, contours, class_value=1):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        # binary mask.
        img_mask = np.zeros(raster_size, np.uint8)
        if contours is None:
            return img_mask
        perim_list, interior_list = contours
        cv2.fillPoly(img_mask, perim_list, class_value)
        cv2.fillPoly(img_mask, interior_list, 0)
        return img_mask

    def _get_polygon_list(self, image_id, class_type):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        df_image = self.df_train_wkt[self.df_train_wkt.ImageId == image_id]
        multipoly_def = df_image[df_image.ClassType == class_type].MultipolygonWKT
        polygonList = None
        if len(multipoly_def) > 0:
            assert len(multipoly_def) == 1
            polygonList = wkt_loads(multipoly_def.values[0])
        return polygonList


if __name__ == "__main__":
    read_train_wkt()
    read_grid_sizes()