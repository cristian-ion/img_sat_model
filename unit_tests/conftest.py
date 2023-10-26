import os

import pytest

from train.image_utils.image_io import image_read


TEST_IMAGES_DIR = "tests/test_images"


def read_image_help(path):
    return image_read(path)


@pytest.fixture()
def images_filelist():
    abs_path = os.path.abspath(TEST_IMAGES_DIR)
    files = {f: os.path.join(abs_path, f) for f in os.listdir(TEST_IMAGES_DIR)}
    return files