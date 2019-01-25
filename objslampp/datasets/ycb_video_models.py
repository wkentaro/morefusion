import pathlib
import shutil
import typing

import gdown

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
