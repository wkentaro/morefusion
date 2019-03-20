import pathlib
import shutil
import typing

import gdown
import numpy as np
import trimesh

from .. import geometry
from .. import sim
from .ycb_video import class_names


class YCBVideoModelsDataset(object):

    root_dir = pathlib.Path.home() / 'data/datasets/YCB/YCB_Video_Models'

    @classmethod
    def download(cls) -> None:
        url: str = 'https://drive.google.com/uc?id=1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu'  # NOQA
        md5: str = 'd3efe74e77fe7d7ca216dde4b7d217fa'

        def postprocess(path: str):
            gdown.extractall(path)
            path_extracted: pathlib.Path = pathlib.Path(path).parent / 'models'
            shutil.move(
                str(path_extracted),
                str(cls.root_dir),
            )

        gdown.cached_download(
            url=url,
            path=str(cls.root_dir) + '.zip',
            md5=md5,
            postprocess=postprocess,
        )

    def __init__(self):
        if not self.root_dir.exists():
            self.download()

    def get_model(
        self,
        class_id: typing.Optional[int] = None,
        class_name: typing.Optional[str] = None,
    ):
        if class_name is None:
            if class_id is None:
                raise ValueError(
                    'either class_id or class_name must not be None'
                )
            else:
                class_name = class_names[class_id]

        return {
            'textured_simple':
                self.root_dir / class_name / 'textured_simple.obj',
            'points_xyz':
                self.root_dir / class_name / 'points.xyz',
        }

    @staticmethod
    def get_spherical_views(visual_file, n_sample=5, radius=0.3):
        eyes = geometry.get_uniform_points_on_sphere(
            n_sample=n_sample, radius=radius
        )
        targets = np.tile([[0, 0, 0]], (len(eyes), 1))

        K, Ts_cam2world, rgbs, depths, segms = sim.pybullet.render_views(
            visual_file, eyes, targets
        )
        return K, Ts_cam2world, rgbs, depths, segms

    @staticmethod
    def get_bbox_diagonal(mesh_file=None, mesh=None):
        if mesh is None:
            mesh = trimesh.load(str(mesh_file), process=False)

        extents = mesh.bounding_box.extents
        bbox_diagonal = np.sqrt((extents ** 2).sum())
        return bbox_diagonal
