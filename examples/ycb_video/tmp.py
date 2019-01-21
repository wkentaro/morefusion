from chainer.backends import cuda
from chainercv.links.model.resnet import ResNet50
import imgviz
import numpy as np
import trimesh

import objslampp


class VoxelMapper(object):

    def __init__(
        self,
        origin=None,
        pitch=None,
        voxel_size=None,
        nchannel=None,
    ):
        self.origin = origin
        self.voxel_size = voxel_size
        self.pitch = pitch
        self.nchannel = nchannel

        self._matrix = None
        self._values = None

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = np.zeros((self.voxel_size,) * 3, dtype=float)
        return self._matrix

    @property
    def values(self):
        if self._values is None:
            self._values = np.zeros(
                (self.voxel_size,) * 3 + (self.nchannel,), dtype=float
            )
        return self._values

    @property
    def voxel_bbox_extents(self):
        return np.array((self.voxel_size * self.pitch,) * 3, dtype=float)

    def add(self, points, values):
        indices = trimesh.voxel.points_to_indices(
            points, self.pitch, self.origin
        )
        keep = ((indices >= 0) & (indices < self.voxel_size)).all(axis=1)
        indices = indices[keep]
        I, J, K = zip(*indices)
        self.matrix[I, J, K] = True
        self.values[I, J, K] = values[keep]

    def as_boxes(self):
        geom = trimesh.voxel.Voxel(
            self.matrix, self.pitch, self.origin
        )
        geom = geom.as_boxes()
        I, J, K = zip(*np.argwhere(self.matrix))
        geom.visual.face_colors = \
            self.values[I, J, K].repeat(12, axis=0)
        return geom

    def as_bbox(self):
        geom = objslampp.vis.trimesh.wired_box(
            self.voxel_bbox_extents,
            translation=self.origin + self.voxel_bbox_extents / 2,
        )
        return geom


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
