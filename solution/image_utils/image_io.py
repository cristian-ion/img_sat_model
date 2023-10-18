import cv2


def read_image(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return None
    print(f"Image size={img.size}, Image shape={img.shape}")
    return img


def save_image(filepath, img):
    cv2.imwrite(filepath, img)
    print(f"Saved image {filepath}")


def show_image(img, title=""):
    cv2.imshow(title, img)
    cv2.waitKey()
