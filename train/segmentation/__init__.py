from train.segmentation.dstl.dataset_dstl import DSTL_NAMECODE, DstlTrainConfig
from train.segmentation.inria.dataset_inria import INRIA_NAMECODE, InriaTrainConfig
from train.segmentation.mu_buildings.dataset_mu_buildings import (
    MU_BUILDINGS_NAMECODE,
    MUBuildingsTrainConfig,
)


def train_config_by_namecode(dataset_namecode: str):
    if dataset_namecode == DSTL_NAMECODE:
        return DstlTrainConfig()
    if dataset_namecode == MU_BUILDINGS_NAMECODE:
        return MUBuildingsTrainConfig()
    if dataset_namecode == INRIA_NAMECODE:
        return InriaTrainConfig()
