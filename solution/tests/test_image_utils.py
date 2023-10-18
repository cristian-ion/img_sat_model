import numpy as np

from solution.image_utils.image_gray import (
    binarize_grayscale,
    convert_numbers_to_bucket,
    extract_red_color_channel,
    rgb_to_gray_simple_average,
)
from solution.image_utils.image_io import read_image


def test_read(images_filelist):
    for image in images_filelist.values():
        img = read_image(image)
        assert img.size > 0


def test_convert_numbers_to_bucket():
    test_out = np.array([30, 127, 200])
    out = convert_numbers_to_bucket(test_out, p=3)
    assert np.array_equal(out, np.array([0, 127, 255], dtype=np.uint8))


def test_simple_average(images_filelist: dict):
    img = read_image(images_filelist["lena.tif"])
    out = rgb_to_gray_simple_average(img)
    assert out[0][0] == (img[0][0][0] / 3 + img[0][0][1] / 3 + img[0][0][2] / 3)


def test_extract_red_color_channel(images_filelist: dict):
    img = read_image(images_filelist["sample_gt_austin1.tif"])
    out = extract_red_color_channel(img)
    assert np.array_equal(out, img[:, :, 2])


def test_binarize_grayscale(images_filelist):
    arr = np.array([10, 127, 128, 255], dtype=np.uint8)
    out = binarize_grayscale(arr, threshold=127)
    assert np.array_equal(out, np.array([0, 0, 1, 1]))
