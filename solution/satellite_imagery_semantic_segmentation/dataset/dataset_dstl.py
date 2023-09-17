from typing import Optional

import numpy as np
import pandas as pd
import torch

from solution.dstl_multiclass_detection.dstl_constants import (
    IMAGEID_COLUMN,
    XMAX_COLUMN,
    YMIN_COLUMN,
)
from solution.dstl_multiclass_detection.dstl_processing import DstlProcessing


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
