import cv2
import numpy as np
from PIL import Image

def predict_mask(img_array: np.ndarray):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = 255 - mask

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return img_array, mask  # img ยังเป็น RGB


def apply_soft_highlight(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_norm = mask / 255.0
    mask_norm = np.expand_dims(mask_norm, axis=-1)

    alpha = 0.8
    beta  = 0.2

    result = img * (beta + alpha * mask_norm)
    return result.astype(np.uint8)


def segment_image(img: Image.Image, img_size=(224, 224)) -> np.ndarray:
    img_array = np.array(img.convert("RGB"))  # PIL → numpy RGB

    img_rgb, mask = predict_mask(img_array)

    if img_rgb is None or mask is None:
        return None

    highlighted = apply_soft_highlight(img_rgb, mask)
    resized = cv2.resize(highlighted, img_size)

    return resized  # (224, 224, 3) uint8