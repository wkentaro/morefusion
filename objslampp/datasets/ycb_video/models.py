import typing

import chainer
import binvox_rw
import gdown
import numpy as np
import path
import trimesh

from ... import extra as extra_module
from ... import utils as utils_module
from ..base import ModelsBase
from .class_names import class_names as ycb_video_class_names


class YCBVideoModels(ModelsBase):

    _root_dir = chainer.dataset.get_dataset_directory(
        'ycb_video/YCB_Video_Models', create_directory=False
    )

    _bbox_diagonal_cache: typing.Dict[str, float] = {}
    _cad_cache: typing.Dict[str, trimesh.Trimesh] = {}
    _pcd_cache: typing.Dict[str, np.ndarray] = {}
    _sdf_cache: typing.Dict[str, typing.Tuple[np.ndarray, np.ndarray]] = {}

    @property
    def class_names(self):
        return ycb_video_class_names

    def get_cad_ids(self, class_id):
        return [self.class_names[class_id]]

    def get_cad_file_from_id(self, cad_id):
        return self.root_dir / cad_id / 'textured_simple.obj'

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

    def get_cad_file(self, class_id):
        class_name = self.class_names[class_id]
        return self.root_dir / class_name / 'textured_simple.obj'

    def get_pcd_file(self, class_id):
        class_name = self.class_names[class_id]
        return self.root_dir / class_name / 'points.xyz'

    def get_sdf(self, class_id):
        class_name = self.class_names[class_id]
        if class_name not in self._sdf_cache:
            points, sdf = self._get_sdf(class_id=class_id)
            self._sdf_cache[class_name] = points, sdf
        return self._sdf_cache[class_name]

    def _get_sdf_file(self, class_id):
        class_name = self.class_names[class_id]
        return self._root_dir / class_name / 'sdf.npz'

    def _get_sdf(self, class_id):
        sdf_file = self._get_sdf_file(class_id=class_id)
        if sdf_file.exists():
            data = np.load(sdf_file)
            points, sdf = data['points'], data['sdf']
        else:
            points = self.get_solid_voxel(class_id=class_id).points
            pitch = self.get_voxel_pitch(32, class_id=class_id)
            points = extra_module.open3d.voxel_down_sample(points, pitch)
            cad = self.get_cad(class_id=class_id)
            sdf = cad.nearest.signed_distance(points)
            np.savez_compressed(sdf_file, points=points, sdf=sdf)
        return points, sdf

    def get_solid_voxel(self, class_id):
        cad_file = self.get_cad_file(class_id=class_id)
        vox_file = utils_module.get_binvox_file(cad_file)
        with open(vox_file, 'rb') as f:
            vox = binvox_rw.read_as_3d_array(f)

        assert vox.dims[0] == vox.dims[1] == vox.dims[2]
        pitch = vox.scale / vox.dims[0]
        voxel = trimesh.voxel.Voxel(
            vox.data,
            pitch=pitch,
            origin=(0.5 * pitch,) * 3 + np.array(vox.translate),
        )
        return voxel

    def get_cad(self, class_id):
        class_name = self.class_names[class_id]
        if class_name not in self._cad_cache:
            cad_file = self.get_cad_file(class_id=class_id)
            cad = trimesh.load(str(cad_file), process=False)
            self._cad_cache[class_name] = cad
        return self._cad_cache[class_name]

    def get_pcd(self, class_id):
        class_name = self.class_names[class_id]
        if class_name not in self._pcd_cache:
            pcd_file = self.get_pcd_file(class_id=class_id)
            pcd = np.loadtxt(pcd_file)
            self._pcd_cache[class_name] = pcd
        return self._pcd_cache[class_name]

    def get_bbox_diagonal(self, class_id):
        class_name = self.class_names[class_id]
        if class_name not in self._bbox_diagonal_cache:
            cad = self.get_cad(class_id=class_id)
            extents = cad.bounding_box.extents
            bbox_diagonal = np.sqrt((extents ** 2).sum())
            self._bbox_diagonal_cache[class_name] = bbox_diagonal
        return self._bbox_diagonal_cache[class_name]

    def get_voxel_pitch(self, dimension, class_id):
        bbox_diagonal = self.get_bbox_diagonal(class_id=class_id)
        return 1. * bbox_diagonal / dimension
