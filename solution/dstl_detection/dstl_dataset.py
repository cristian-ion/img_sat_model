import torch
import pandas as pd
import numpy as np

from solution.dstl_detection.dstl_processing import DstlProcessing
from solution.dstl_detection.dstl_constants import COL_IMAGEID


class DstlDataset(torch.utils.data.Dataset):
    def __init__(self, transform, train_file, grid_sizes_file, classes, train_res_y, train_res_x) -> None:
        super().__init__()

        df_wkt = pd.read_csv(train_file)
        df_gs = pd.read_csv(grid_sizes_file, names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

        # self.df_samples = df_wkt
        self.transform = transform
        self.processing = DstlProcessing(df_wkt, df_gs, classes)
        self.df_wkt = df_wkt
        self.train_res_y = train_res_y
        self.train_res_x = train_res_x

    def __len__(self):
        return len(self.df_wkt)

    def __getitem__(self, index):
        sample = self.df_wkt.iloc[index]

        image, mask = self.processing.read_image_and_mask(
            raster_size=(self.train_res_y, self.train_res_x),
            image_id=sample[COL_IMAGEID],
        )
        # mask = mask[0]

        mask[mask == 255] = 1
        mask = mask.astype(np.float32)

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        mask = torch.from_numpy(mask)
        return image, mask
