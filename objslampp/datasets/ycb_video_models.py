import pathlib
import shutil
import typing

import gdown
import numpy as np

from .. import geometry
from .. import sim
from .ycb import class_id_to_name


class YCBVideoModelsDataset(object):

    root_dir = pathlib.Path.home() / 'data/datasets/YCB/YCB_Video_Models'

    @classmethod
    def download(cls) -> None:
        url: str = 'https://drive.google.com/uc?id=1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu'  # NOQA
        md5: str = 'd3efe74e77fe7d7ca216dde4b7d217fa'

        def postprocess(path: pathlib.Path):
            gdown.extractall(path)
            path_extracted: pathlib.Path = path.parent / 'models'
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
                class_name = class_id_to_name[class_id]

        return {
            'textured_simple':
                self.root_dir / class_name / 'textured_simple.obj',
        }

    def get_spherical_views(self, visual_file, n_sample=5, radius=0.3):
        eyes = geometry.get_uniform_points_on_sphere(
            n_sample=n_sample, radius=radius
        )
        targets = np.tile([[0, 0, 0]], (len(eyes), 1))

        views = sim.pybullet.render_views(visual_file, eyes, targets)
        rgbs, depths, segms = zip(*views)

        Ts_cam2world = [
            geometry.look_at(eye, target, up=[0, -1, 0])
            for eye, target in zip(eyes, targets)
        ]

        return Ts_cam2world, rgbs, depths, segms
