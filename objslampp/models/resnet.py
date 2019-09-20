import chainer
import numpy as np

from chainercv2.model_provider import get_model


class ResNet18Extractor(chainer.Chain):

    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    def __init__(self, unchain_at='res2'):
        assert unchain_at == 'res2'
        self._unchain_at = unchain_at

        super().__init__()
        with self.init_scope():
            model = get_model('resnet18', pretrained=True)
            self.init_block = model.features.init_block
            self.res2 = model.features.stage1
            self.res3 = model.features.stage2
            self.res4 = model.features.stage3
            self.res4.unit1.body.conv1.conv.stride = (1, 1)
            self.res4.unit1.identity_conv.conv.stride = (1, 1)
            self.res4.unit2.body.conv1.conv.dilate = (2, 2)
            self.res4.unit2.body.conv1.conv.pad = (2, 2)
            self.res4.unit2.body.conv2.conv.dilate = (2, 2)
            self.res4.unit2.body.conv2.conv.pad = (2, 2)
            self.res5 = model.features.stage4
            self.res5.unit1.body.conv1.conv.stride = (1, 1)
            self.res5.unit1.identity_conv.conv.stride = (1, 1)
            self.res5.unit2.body.conv1.conv.dilate = (4, 4)
            self.res5.unit2.body.conv1.conv.pad = (4, 4)
            self.res5.unit2.body.conv2.conv.dilate = (4, 4)
            self.res5.unit2.body.conv2.conv.pad = (4, 4)

        self.mean = np.array(self.mean_rgb, dtype=np.float32)[:, None, None]
        self.std = np.array(self.std_rgb, dtype=np.float32)[:, None, None]

    def __call__(self, x):
        self.mean = self.xp.asarray(self.mean)
        self.std = self.xp.asarray(self.std)
        h = (x / 255. - self.mean[None]) / self.std[None]
        with chainer.using_config('train', False):  # disable update bn
            h = self.init_block(h)
            h = self.res2(h)
            if self._unchain_at == 'res2':
                h.unchain()
            h = self.res3(h)
            h = self.res4(h)
            h = self.res5(h)
        return h
