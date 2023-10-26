import cv2
import numpy as np

from train.image_utils.image_io import image_save

SAMPLE_IMAGE = "lena.tif"
OUTDIR = "l2_out"


# References:
# https://tannerhelland.com/2011/10/01/grayscale-image-algorithm-vb6.html
# https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html


average_weights = {
    1: [0.11, 0.59, 0.3],
    2: [0.0722, 0.2126, 0.7152],
    3: [0.114, 0.587, 0.299],
}


def _check_image(img):
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    assert np.min(img) >= 0
    assert np.max(img) < 256


def rgb_to_gray_simple_average(img, save=False):
    _check_image(img)
    out = np.mean(img, axis=2)
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_simple_average.png", out)
    return out


def weighted_average_1(img, save=False):
    _check_image(img)
    out = np.average(img, axis=2, weights=average_weights[1])
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_weighted_average_1.png", out)
    return out


def weighted_average_2(img, save=False):
    _check_image(img)
    out = np.average(img, axis=2, weights=average_weights[2])
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_weighted_average_2.png", out)
    return out


def weighted_average_3(img, save=False):
    _check_image(img)
    out = np.average(img, axis=2, weights=average_weights[3])
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_weighted_average_3.png", out)
    return out


def desaturation(img, save=False):
    _check_image(img)
    print(img)
    # print("min")
    # print(np.min(img, axis=2))
    # print("max")
    # print(np.max(img, axis=2))
    out = np.min(img, axis=2) / 2 + np.max(img, axis=2) / 2
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_desaturation.png", out)
    return out


def decomposition_min(img, save=False):
    _check_image(img)
    out = np.min(img, axis=2)
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_decomposition_min.png", out)
    return out


def decomposition_max(img, save=False):
    _check_image(img)
    out = np.max(img, axis=2)
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_decomposition_max.png", out)
    return out


def extract_red_color_channel(img, save=False):
    _check_image(img)
    out = img[:, :, 2]
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_single_colour_channel_r.png", out)
    return out


def single_colour_channel_g(img, save=False):
    _check_image(img)
    out = img[:, :, 1]
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_single_colour_channel_g.png", out)
    return out


def single_colour_channel_b(img, save=False):
    _check_image(img)
    out = img[:, :, 0]
    print(out.shape)
    if save:
        image_save(OUTDIR + "/lena_single_colour_channel_b.png", out)
    return out


def convert_numbers_to_bucket(out, p):
    assert p >= 2
    assert p <= 256
    conversion_factor = 255 / (p - 1)
    out = np.rint(out / conversion_factor).astype(np.uint8) * conversion_factor
    return out.astype(np.uint8)


def custom_number_of_grey_shades(img, p, save=False):
    """
    https://tannerhelland.com/2011/10/01/grayscale-image-algorithm-vb6.html
    img: Color image (h, w, c)
    p: Number of grey shades < 256

    > p = 3
    > conversion_factor = 255 / 2 = 127.5
    >
    """
    _check_image(img)
    out = weighted_average_2(img, save=False)
    out = convert_numbers_to_bucket(out, p)
    print(out.shape)
    print(out.dtype)
    if save:
        image_save(OUTDIR + "/lena_custom_number_of_grey_channels.png", out)
    return out


def nearest_color_2(color):
    avg = (color[0] + color[1] + color[2]) / 3
    if avg >= 128:
        return 255
    return 0


def nearest_color_p(color, p):
    avg = (color[0] + color[1] + color[2]) / 3
    return convert_numbers_to_bucket(avg, p)


def propagate_error_floyd_steinberg(i, j, img, err):
    m = img.shape[0]
    n = img.shape[1]
    if i >= m and j >= n:
        return

    if j + 1 < n:
        img[i][j + 1] += err * (7 / 16)

    if i + 1 >= m:
        return

    img[i + 1][j - 1] += err * (3 / 16)
    img[i + 1][j] += err * (5 / 16)
    if j + 1 < n:
        img[i + 1][j + 1] += err * (1 / 16)


def propagate_error_burkes(i, j, img, err):
    m = img.shape[0]
    n = img.shape[1]
    if (i >= m or i < 0) and (j >= n or j < 0):
        return

    # current line
    if j + 1 < n:
        img[i][j + 1] += err * (8 / 32)
    if j + 2 < n:
        img[i][j + 2] += err * (4 / 32)

    # next line
    if i + 1 >= m:
        return

    if j - 1 >= 0:
        img[i + 1][j - 1] += err * (2 / 32)
    if j - 2 >= 0:
        img[i + 1][j - 2] += err * (4 / 32)

    img[i + 1][j] += err * (8 / 32)

    if j + 1 < n:
        img[i + 1][j + 1] += err * (4 / 32)
    if j + 2 < n:
        img[i + 1][j + 2] += err * (2 / 32)


def custom_number_of_grey_shades_with_error_diffusion_dithering(
    img,
    save=False,
    error_prop=propagate_error_burkes,
    p=4,
):
    # store only two rows of the image
    _check_image(img)
    out = np.zeros((img.shape[0], img.shape[1]), dtype=int)
    m = img.shape[0]
    n = img.shape[1]

    img = img.astype(np.float32)

    for i in range(m):
        for j in range(n):
            if p == 2:
                out[i][j] = nearest_color_2(img[i][j])
            else:
                out[i][j] = nearest_color_p(img[i][j], p)
            err = img[i][j] - out[i][j]
            error_prop(i, j, img, err)

    print(out.shape)
    if save:
        image_save(
            OUTDIR
            + f"/lena_custom_number_of_grey_shades_with_error_diffusion_dithering_{error_prop.__name__}_p{p}.png",
            out,
        )
    return out


def test_convert_numbers_to_bucket():
    test_out = np.array([30, 127, 200])
    assert np.array_equal(
        convert_numbers_to_bucket(test_out, p=3),
        np.array([0, 127, 255], dtype=np.uint8),
    )


def test_nearest_color():
    assert nearest_color_2(np.array([127, 127, 127])) == 0
    assert nearest_color_2(np.array([128, 128, 128])) == 255


def grayscale_to_rgb(gray, save=False):
    # todo: de prezentat
    m = gray.shape[0]
    n = gray.shape[1]
    img = np.zeros((m, n, 3), dtype=np.uint8)

    w = [0.11, 0.59, 0.3]

    for i in range(m):
        for j in range(n):
            img[i][j][0] = w[0] * gray[i][j]
            img[i][j][1] = w[1] * gray[i][j]
            img[i][j][2] = w[2] * gray[i][j]

    if save:
        image_save(OUTDIR + "/lena_grayscale_to_rgb.png", img)
    return img


def binarize_grayscale(gray, threshold=127):
    return np.where(gray > threshold, 1.0, 0.0).astype(np.float32)


def grayscale_resize_nearest_uint8(img, new_w, new_h):
    return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)


def probability_to_black_and_white_uint8(preds):
    return np.where(preds > 0.5, 255, 0).astype(np.uint8)


def gray_nearest_black_and_white_uint8(gray):
    return np.where(gray > 127, 255, 0).astype(np.uint8)
