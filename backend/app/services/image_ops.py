"""
Image preprocessing for the face-shape models in plain NumPy (no TensorFlow).

The models were trained on Kaggle with TensorFlow:

    img = tf.image.resize_with_pad(img, 224, 224)      # bilinear, half-pixel centres, no antialias
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = keras.applications.resnet50.preprocess_input(img)

`resize_with_pad` and `apply_preprocess` below reproduce those steps exactly,
so the server needs only NumPy + LiteRT at inference time. The Kaggle script
training/convert_tflite.py checks this file against TensorFlow on the real
test images before the models are shipped.
"""
import math

import numpy as np

IMAGENET_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """tf.image.resize(method="bilinear", antialias=False): half-pixel centres, edge clamping."""
    in_h, in_w = img.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return img.astype(np.float32, copy=True)

    def axis(out_size, in_size):
        scale = in_size / out_size
        src = (np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5
        lo_f = np.floor(src)
        frac = (src - lo_f).astype(np.float32)
        lo = np.clip(lo_f, 0, in_size - 1).astype(np.int64)
        hi = np.clip(lo_f + 1, 0, in_size - 1).astype(np.int64)
        return lo, hi, frac

    y0, y1, fy = axis(out_h, in_h)
    x0, x1, fx = axis(out_w, in_w)
    img = img.astype(np.float32, copy=False)
    top = img[y0][:, x0] * (1 - fx)[None, :, None] + img[y0][:, x1] * fx[None, :, None]
    bottom = img[y1][:, x0] * (1 - fx)[None, :, None] + img[y1][:, x1] * fx[None, :, None]
    return top * (1 - fy)[:, None, None] + bottom * fy[:, None, None]


def resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """tf.image.resize_with_pad: keep the aspect ratio, resize, pad the rest with zeros (centred)."""
    h, w = img.shape[:2]
    ratio = max(w / target_w, h / target_h)
    new_h = int(math.floor(h / ratio))
    new_w = int(math.floor(w / ratio))
    resized = _resize_bilinear(img, new_h, new_w)
    out = np.zeros((target_h, target_w, img.shape[2]), dtype=np.float32)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


def to_uint8_range(img: np.ndarray) -> np.ndarray:
    """Training stored the resized image as uint8 (round + clip) before preprocessing."""
    return np.clip(np.round(img), 0, 255).astype(np.float32)


def apply_preprocess(arr, name):
    """arr: (..., H, W, 3) float32 RGB 0-255 -> what the backbone was trained on."""
    if name == "resnet50":   # keras.applications.resnet50.preprocess_input ("caffe" mode)
        out = arr[..., ::-1].astype(np.float32, copy=True)
        out -= IMAGENET_BGR_MEAN
        return out
    if name in ("none", "", None):
        return arr
    raise ValueError(f"unknown preprocess {name!r}")


def prepare_batch(rgb: np.ndarray, img_size, preprocess: str, tta: bool) -> np.ndarray:
    """(H, W, 3) RGB 0-255 -> (1 or 2, h, w, 3) model input; with tta the second row is the mirror image."""
    x = to_uint8_range(resize_with_pad(rgb, img_size[0], img_size[1]))
    batch = np.stack([x, x[:, ::-1]]) if tta else x[None]
    return apply_preprocess(batch, preprocess)
