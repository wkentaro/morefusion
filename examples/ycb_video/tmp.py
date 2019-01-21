from chainer.backends import cuda
from chainercv.links.model.resnet import ResNet50
import imgviz
import numpy as np


class ResNetFeatureExtractor(object):

    def __init__(self, gpu=0):
        if gpu >= 0:
            cuda.get_device_from_id(gpu).use()
        self._resnet = ResNet50(pretrained_model='imagenet', arch='he')
        self._resnet.pick = ['res4']
        if gpu >= 0:
            self._resnet.to_gpu()
        self._nchannel2rgb = imgviz.Nchannel2RGB()

    def extract_feature(self, rgb):
        x = rgb.transpose(2, 0, 1)
        x = x - self._resnet.mean
        x = x[None]
        if self._resnet.xp != np:
            x = cuda.to_gpu(x)
        feat, = self._resnet(x)
        feat = cuda.to_cpu(feat[0].array)
        return feat.transpose(1, 2, 0)

    def feature2rgb(self, feat, mask_fg):
        dst = self._nchannel2rgb(feat, dtype=float)
        H, W = mask_fg.shape[:2]
        dst = imgviz.resize(dst, height=H, width=W)
        dst = (dst * 255).astype(np.uint8)
        dst[~mask_fg] = 0
        return dst
