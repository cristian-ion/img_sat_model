import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def draw_things(img, masks, draw_masks=True, draw_boxes=False):
    """
    Known problems:
    File "/Users/cristianion/Desktop/img_sat_model/train/segmentation/train_val.py", line 279, in <listcomp>
        draw_things(img, tmp)
    File "/Users/cristianion/Desktop/img_sat_model/train/segmentation/train_val.py", line 43, in draw_things
        boxes=masks_to_boxes(mask),
            ^^^^^^^^^^^^^^^^^^^^
    File "/Users/cristianion/Desktop/img_sat_model/.venv/lib/python3.11/site-packages/torchvision/ops/boxes.py", line 412, in masks_to_boxes
        bounding_boxes[index, 0] = torch.min(x)
                                ^^^^^^^^^^^^
    """
    print(f"img.shape={img.shape}")
    print(f"masks.shape={masks.shape}")
    print(masks.numel())
    canvas = img
    if draw_masks:
        canvas = draw_segmentation_masks(
            image=canvas, masks=masks, alpha=0.7, colors="red"
        )
    if draw_boxes:
        canvas = draw_bounding_boxes(
            image=canvas,
            boxes=masks_to_boxes(masks),
            colors="red",
        )
    return canvas


def plot_img(imgs, fname):
    #
    # https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#semantic-segmentation-models
    #
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(fname=fname, dpi=400)
