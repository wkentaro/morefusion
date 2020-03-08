# https://github.com/knorth55/chainer-dense-fusion/blob/44d577495acc839aaecf4811e7d6eccf19482f8e/chainer_dense_fusion/links/model/resnet.py  # NOQA
import chainer
import chainer.functions as F
import chainer.links as L
import chainercv
import numpy as np


class ResNet(chainercv.links.PickableSequentialChain):

    _blocks = {
        13: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
    }

    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    def __init__(self, n_layer):
        blocks = self._blocks[n_layer]
        super().__init__()
        with self.init_scope():
            # 1/1 -> 1/2
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3, nobias=True)
            # ceil_mode=False in pytorch corresponds to cover_all=False
            # 1/2 -> 1/4
            self.pool1 = lambda x: F.max_pooling_2d(
                x, ksize=3, stride=2, pad=1, cover_all=False
            )
            self.res2 = ResBlock(blocks[0], 64, 64, 1, 1, residual_conv=False)
            # 1/4 -> 1/8
            self.res3 = ResBlock(blocks[1], 64, 128, 2, 1)
            self.res4 = ResBlock(blocks[2], 128, 256, 1, 2)
            self.res5 = ResBlock(blocks[3], 256, 512, 1, 4)

        self.mean = np.array(self.mean_rgb, dtype=np.float32)[:, None, None]
        self.std = np.array(self.std_rgb, dtype=np.float32)[:, None, None]

    def __call__(self, x):
        self.mean = self.xp.asarray(self.mean)
        self.std = self.xp.asarray(self.std)
        h = (x / 255.0 - self.mean[None]) / self.std[None]
        return super().__call__(h)


class ResNet18(ResNet):
    def __init__(self):
        super().__init__(n_layer=13)


class ResNet34(ResNet):
    def __init__(self):
        super().__init__(n_layer=34)


class ResBlock(chainer.Chain):
    def __init__(
        self,
        n_layer,
        in_channels,
        out_channels,
        stride,
        dilate,
        residual_conv=True,
    ):
        super().__init__()
        with self.init_scope():
            self.a = BasicBlock(
                in_channels,
                out_channels,
                stride,
                1,
                residual_conv=residual_conv,
            )
            for i in range(n_layer - 1):
                name = "b{}".format(i + 1)
                block = BasicBlock(
                    out_channels, out_channels, 1, dilate, residual_conv=False
                )
                setattr(self, name, block)
        self.n_layer = n_layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(self.n_layer - 1):
            h = getattr(self, "b{}".format(i + 1))(h)
        return h


class BasicBlock(chainer.Chain):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dilate,
        initialW=None,
        residual_conv=False,
    ):
        super().__init__()
        with self.init_scope():
            # pad = dilate
            self.conv1 = L.Convolution2D(
                in_channels,
                out_channels,
                3,
                stride,
                pad=dilate,
                dilate=dilate,
                nobias=True,
            )
            self.conv2 = L.Convolution2D(
                out_channels,
                out_channels,
                3,
                1,
                pad=dilate,
                dilate=dilate,
                nobias=True,
            )
            if residual_conv:
                self.residual_conv = L.Convolution2D(
                    in_channels, out_channels, 1, stride, nobias=True
                )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)

        if hasattr(self, "residual_conv"):
            residual = self.residual_conv(x)
        else:
            residual = x
        h = h + residual
        h = F.relu(h)
        return h
