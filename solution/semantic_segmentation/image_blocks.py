import math

import cv2
import numpy as np


def image_split_in_blocks(img, block_size, new_size=None, use_padding=False):
    blocks = []
    h = img.shape[0]
    w = img.shape[1]

    if use_padding:
        new_size = (h / block_size) * block_size
        rest = h % block_size
        if rest > 0:
            new_size += block_size

        canvas = np.zeros((int(new_size), int(new_size), 3))
        canvas[0:h, 0:w] = img
    else:
        canvas = img
        if new_size:
            p2 = new_size
        else:
            p2 = int(math.pow(2, int(math.ceil(math.log2(h))) - 1))
        canvas = cv2.resize(canvas, (p2, p2), interpolation=cv2.INTER_AREA)
        print(canvas.shape)

    for y0 in range(0, canvas.shape[0], block_size):
        for x0 in range(0, canvas.shape[1], block_size):
            block = np.zeros((block_size, block_size, 3), dtype="uint8")
            block[0:block_size, 0:block_size] = canvas[
                y0 : (y0 + block_size), x0 : (x0 + block_size)
            ]
            blocks.append(block)

    print(len(blocks))
    return blocks


def image_join_blocks(blocks):
    if not blocks:
        return None

    block_size = blocks[0].shape[0]

    w = int(math.sqrt(len(blocks))) * block_size
    h = w

    img = np.zeros((h, w, 3), dtype="uint8")
    current_block = 0
    for y0 in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            img[y0 : (y0 + block_size), x0 : (x0 + block_size)] = blocks[current_block]
            current_block += 1

    return img


if __name__ == "__main__":
    img = cv2.imread("lena.tif")
    print(img.size)
    print(img.shape)

    height = img.shape[0]
    width = img.shape[1]

    dim = (5000, 5000)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print(img.size)
    print(img.shape)

    cv2.imshow("lena", img)
    cv2.waitKey()

    blocks = image_split_in_blocks(img, 512, new_size=4096)

    for block in blocks:
        cv2.imshow("lena", block)
        cv2.waitKey()

    img = image_join_blocks(blocks)
    cv2.imshow("lena", img)
    cv2.waitKey()
