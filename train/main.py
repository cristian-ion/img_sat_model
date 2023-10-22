import sys

from train.segmentation.dstl.dataset_dstl import DSTL_NAMECODE
from train.segmentation.inria.dataset_inria import INRIA_NAMECODE
from train.segmentation.mu_buildings.dataset_mu_buildings import MU_BUILDINGS_NAMECODE
from train.segmentation.train import Train


def train():
    print(sys.argv)
    if len(sys.argv) != 2:
        print(
            f"Please provide dataset namecode: {MU_BUILDINGS_NAMECODE}, {DSTL_NAMECODE}, {INRIA_NAMECODE}."
        )
        sys.exit(0)

    backprop = Train(sys.argv[1])
    backprop.train()


if __name__ == "__main__":
    train()
