import cv2


def image_read(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return None
    print(f"Image size={img.size}, Image shape={img.shape}")
    return img


def image_save(filepath, img):
    cv2.imwrite(filepath, img)
    print(f"Saved image {filepath}")


def image_show(img, title=""):
    cv2.imshow(title, img)
    cv2.waitKey()
