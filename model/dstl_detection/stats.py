import numpy as np
import pandas as pd

from model.dstl_detection.dstl_constants import *


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