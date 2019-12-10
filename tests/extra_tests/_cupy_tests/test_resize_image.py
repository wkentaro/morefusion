import unittest

from chainer.backends import cuda
from chainer import testing
import imgviz
import numpy as np
import scipy.misc

from morefusion.extra._cupy import resize_image


class TestResizeImage(unittest.TestCase):

    def setUp(self):
        self.image_uint8 = scipy.misc.face()
        self.image_float32 = self.image_uint8.astype(np.float32) / 255
        self.image_bool = imgviz.rgb2gray(self.image_uint8) > 127

    def check_resize_image_uint8(self, src):
        H, W, C = src.shape
        H_2x, W_2x = 2 * H, 2 * W
        dst = resize_image(src, (H_2x, W_2x), order='HWC')
        assert dst.shape == (H_2x, W_2x, C)
        assert src.dtype == dst.dtype

        dst = resize_image(dst, (H, W), order='HWC')
        assert dst.shape == (H, W, C)
        assert src.dtype == dst.dtype

        src = cuda.to_cpu(src)
        dst = cuda.to_cpu(dst)
        diff = np.abs(src.astype(float) / 255 - dst.astype(float) / 255)
        assert diff.min() == 0
        assert diff.mean() < 0.005
        assert diff.max() < 0.1

    def check_resize_image_float32(self, src):
        H, W, C = src.shape
        H_2x, W_2x = 2 * H, 2 * W
        dst = resize_image(src, (H_2x, W_2x), order='HWC')
        assert dst.shape == (H_2x, W_2x, C)
        assert src.dtype == dst.dtype

        dst = resize_image(dst, (H, W), order='HWC')
        assert dst.shape == (H, W, C)
        assert src.dtype == dst.dtype

        src = cuda.to_cpu(src)
        dst = cuda.to_cpu(dst)
        diff = np.abs(src.astype(float) - dst.astype(float))
        assert diff.min() == 0
        assert diff.mean() < 0.005
        assert diff.max() < 0.1

    def check_resize_image_bool(self, src):
        H, W = src.shape
        H_2x, W_2x = 2 * H, 2 * W
        dst = resize_image(src, (H_2x, W_2x), order='HW')
        assert dst.shape == (H_2x, W_2x)
        assert src.dtype == dst.dtype

        dst = resize_image(dst, (H, W), order='HW')
        assert dst.shape == (H, W)
        assert src.dtype == dst.dtype

        src = cuda.to_cpu(src)
        dst = cuda.to_cpu(dst)
        np.testing.assert_allclose(src, dst)

    def check_resize_image(self, src):
        if src.dtype == np.uint8:
            self.check_resize_image_uint8(src)
        elif src.dtype == np.float32:
            self.check_resize_image_float32(src)
        elif src.dtype == bool:
            self.check_resize_image_bool(src)
        else:
            raise ValueError()

    def test_resize_image_cpu(self):
        self.check_resize_image(self.image_uint8)
        self.check_resize_image(self.image_float32)
        self.check_resize_image(self.image_bool)

    @testing.attr.gpu
    def test_resize_image_gpu(self):
        self.check_resize_image(cuda.to_gpu(self.image_uint8))
        self.check_resize_image(cuda.to_gpu(self.image_float32))
        self.check_resize_image(cuda.to_gpu(self.image_bool))
