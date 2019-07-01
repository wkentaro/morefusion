import typing

import chainer
import gdown
import numpy as np
import path
import trimesh

from ..base import DatasetBase
from .class_names import class_names


class YCBVideoModels(DatasetBase):

    _root_dir = chainer.dataset.get_dataset_directory(
        'ycb_video/YCB_Video_Models', create_directory=False
    )
    _class_names = class_names

    _cad_cache: typing.Dict[str, trimesh.Trimesh] = {}
    _pcd_cache: typing.Dict[str, np.ndarray] = {}

    def __len__(self):
        raise NotImplementedError

    def split(self):
        raise NotImplementedError

    def ids(self):
        raise NotImplementedError

    @property
    def class_names(self):
        return self._class_names

    @property
    def n_class(self):
        return len(self.class_names)

    @classmethod
    def download(cls) -> None:
        url: str = 'https://drive.google.com/uc?id=1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu'  # NOQA
        md5: str = 'd3efe74e77fe7d7ca216dde4b7d217fa'

        def postprocess(file: str):
            gdown.extractall(file)
            file_extracted = path.Path(file).parent / 'models'
            file_extracted.move(cls._root_dir)

        gdown.cached_download(
            url=url,
            path=cls._root_dir + '.zip',
            md5=md5,
            postprocess=postprocess,
        )

    def __init__(self):
        if not self.root_dir.exists():
            self.download()

    def _get_class_name(self, class_id=None, class_name=None):
        if class_name is None:
            if class_id is None:
                raise ValueError(
                    'either class_id or class_name must not be None'
                )
            else:
                class_name = class_names[class_id]
        return class_name

    def get_model_files(self, class_id=None, class_name=None):
        class_name = self._get_class_name(
            class_id=class_id, class_name=class_name
        )
        return {
            'textured_simple':
                self.root_dir / class_name / 'textured_simple.obj',
            'points_xyz':
                self.root_dir / class_name / 'points.xyz',
        }

    def get_cad_file(self, *args, **kwargs):
        return self.get_model_files(*args, **kwargs)['textured_simple']

    def get_pcd_file(self, *args, **kwargs):
        return self.get_model_files(*args, **kwargs)['points_xyz']

    def get_cad(self, *args, **kwargs):
        class_name = self._get_class_name(*args, **kwargs)
        if class_name not in self._cad_cache:
            cad_file = self.get_cad_file(*args, **kwargs)
            cad = trimesh.load(str(cad_file), process=False)
            self._cad_cache[class_name] = cad
        return self._cad_cache[class_name]

    def get_pcd(self, *args, **kwargs):
        class_name = self._get_class_name(*args, **kwargs)
        if class_name not in self._pcd_cache:
            pcd_file = self.get_pcd_file(*args, **kwargs)
            pcd = np.loadtxt(pcd_file)
            self._pcd_cache[class_name] = pcd
        return self._pcd_cache[class_name]

    def get_bbox_diagonal(self, *args, **kwargs):
        cad = self.get_cad(*args, **kwargs)
        extents = cad.bounding_box.extents
        bbox_diagonal = np.sqrt((extents ** 2).sum())
        return bbox_diagonal

    def get_voxel_pitch(self, dimension, *args, **kwargs):
        bbox_diagonal = self.get_bbox_diagonal(*args, **kwargs)
        return 1. * bbox_diagonal / dimension
