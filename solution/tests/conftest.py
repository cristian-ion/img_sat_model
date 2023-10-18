import os

import pytest

from solution.image_utils.image_io import read_image

TEST_LENA_IMAGE = "test_images/lena.tif"
TEST_IMAGES_DIR = "solution/tests/test_images"


def read_image_help(path):
    return read_image(path)


@pytest.fixture()
def images_filelist():
    abs_path = os.path.abspath(TEST_IMAGES_DIR)
    files = {f: os.path.join(abs_path, f) for f in os.listdir(TEST_IMAGES_DIR)}
    return files
