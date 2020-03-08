# flake8: noqa
# https://github.com/knorth55/chainer-dense-fusion/blob/master/chainer_dense_fusion/links/model/pspnet.py  # NOQA

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class PSPNetExtractor(chainer.Chain):
    def __init__(self):
        super(PSPNetExtractor, self).__init__()
        sizes = [1, 2, 3, 6]
        with self.init_scope():
            self.psp = PSPModule(512, 1024, sizes)
            # 1/8 -> 1/4
            self.up1 = PSPUpsample(1024, 256)
            # 1/4 -> 1/2
            self.up2 = PSPUpsample(256, 64)
            # 1/2 -> 1
            self.up3 = PSPUpsample(64, 64)
            self.conv1 = L.Convolution2D(64, 32, 1)

    def __call__(self, x):
        # psp module
        h = self.psp(x)
        h = F.dropout(h, 0.3)
        # upsample
        h = F.dropout(self.up1(h), 0.15)
        h = F.dropout(self.up2(h), 0.15)
        h = self.up3(h)
        # head
        h = self.conv1(h)
        feat = F.log_softmax(h)
        return feat


class PSPModule(chainer.Chain):
    def __init__(self, in_channels, out_channels, sizes):
        super(PSPModule, self).__init__()
        with self.init_scope():
            for i in range(len(sizes)):
                setattr(
                    self,
                    "conv{}".format(i + 1),
                    L.Convolution2D(in_channels, in_channels, 1, nobias=True),
                )
            self.bottleneck = L.Convolution2D(
                in_channels * (len(sizes) + 1), out_channels, 1
            )
        self.sizes = sizes

    def __call__(self, x):
        H, W = x.shape[2:]
        kh = H // np.array(self.sizes)
        kw = W // np.array(self.sizes)
        ksizes = list(zip(kh, kw))

        # extract
        hs = []
        for i, ksize in enumerate(ksizes):
            h = F.average_pooling_2d(x, ksize, ksize)
            h = getattr(self, "conv{}".format(i + 1))(h)
            h = F.resize_images(h, (H, W))
            hs.append(h)
        hs.append(x)
        h = F.relu(self.bottleneck(F.concat(hs, axis=1)))
        return h


class PSPUpsample(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, 3, 1, pad=1)
            self.prelu = L.PReLU()

    def __call__(self, x):
        H, W = x.shape[2:]
        h = F.resize_images(x, (H * 2, W * 2))
        h = self.prelu(self.conv(h))
        return h
