"""
image_ops.py must give the same numbers as the TensorFlow ops the models were trained with
(tf.image.resize_with_pad + round to uint8 + resnet50 preprocess_input). TensorFlow isn't
installed on the server, so these tests compare against a slow, literal re-statement of
TF's ResizeBilinear (half_pixel_centers=True); training/convert_tflite.py also checks the
module against real TensorFlow on the test images.
Run: python -m unittest discover -s tests
"""
import math
import unittest

import numpy as np

from app.services import image_ops


def tf_resize_bilinear_reference(img, out_h, out_w):
    in_h, in_w = img.shape[:2]
    out = np.zeros((out_h, out_w, img.shape[2]))
    for y in range(out_h):
        sy = (y + 0.5) * in_h / out_h - 0.5
        ly, uy, fy = max(math.floor(sy), 0), min(math.ceil(sy), in_h - 1), sy - math.floor(sy)
        for x in range(out_w):
            sx = (x + 0.5) * in_w / out_w - 0.5
            lx, ux, fx = max(math.floor(sx), 0), min(math.ceil(sx), in_w - 1), sx - math.floor(sx)
            top = img[ly, lx] * (1 - fx) + img[ly, ux] * fx
            bottom = img[uy, lx] * (1 - fx) + img[uy, ux] * fx
            out[y, x] = top * (1 - fy) + bottom * fy
    return out


class Resize(unittest.TestCase):
    def test_bilinear_matches_tf_definition(self):
        rng = np.random.default_rng(0)
        for shape, out in (((37, 53, 3), (20, 17)), ((64, 64, 3), (45, 45)), ((9, 9, 3), (23, 31))):
            img = rng.uniform(0, 255, shape).astype(np.float32)
            got = image_ops._resize_bilinear(img, *out)
            np.testing.assert_allclose(got, tf_resize_bilinear_reference(img.astype(np.float64), *out), atol=1e-3)

    def test_square_crop_is_not_padded(self):
        img = np.full((320, 320, 3), 200, np.float32)
        out = image_ops.resize_with_pad(img, 224, 224)
        self.assertEqual(out.shape, (224, 224, 3))
        self.assertTrue(np.allclose(out, 200))

    def test_tall_image_is_padded_left_and_right(self):
        img = np.full((300, 200, 3), 50, np.float32)
        out = image_ops.resize_with_pad(img, 224, 224)
        # ratio 300/224 -> 224 x 149, offset (224-149)//2 = 37 like tf.image.pad_to_bounding_box
        self.assertTrue(np.all(out[:, :37] == 0) and np.all(out[:, 186:] == 0))
        self.assertTrue(np.allclose(out[:, 37:186], 50))


class Batch(unittest.TestCase):
    def test_tta_row_is_mirror_and_preprocess_is_caffe(self):
        rng = np.random.default_rng(1)
        img = rng.uniform(0, 255, (224, 224, 3)).astype(np.float32)
        batch = image_ops.prepare_batch(img, (224, 224), "resnet50", tta=True)
        self.assertEqual(batch.shape, (2, 224, 224, 3))
        np.testing.assert_allclose(batch[1], batch[0][:, ::-1])
        rounded = np.round(img)
        np.testing.assert_allclose(batch[0, 5, 7], rounded[5, 7, ::-1] - image_ops.IMAGENET_BGR_MEAN, rtol=1e-6)
        self.assertEqual(image_ops.prepare_batch(img, (224, 224), "none", tta=False).shape, (1, 224, 224, 3))


if __name__ == "__main__":
    unittest.main()
