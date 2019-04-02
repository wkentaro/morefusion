import pathlib
import typing

import chainer
import gdown
import imgviz
import numpy as np
import scipy.io

from ..base import DatasetBase


class YCBVideoDataset(DatasetBase):

    _root_dir = chainer.dataset.get_dataset_directory(
        'ycb_video/YCB_Video_Dataset', create_directory=False
    )
    _root_dir = pathlib.Path(_root_dir)
    _data_dir = 'data'

    def __init__(self, split: str, sampling=1):
        assert split in ('train', 'val', 'trainval', 'keyframe')
        self._split = split
        self._ids = self.get_ids(sampling=sampling)

        if not self.root_dir.exists():
            self.download()

    def get_example(self, i):
        image_id = self.ids[i]
        return self.get_frame(image_id)

    @staticmethod
    def get_image_id(
        scene_id: typing.Union[int, str],
        frame_id: typing.Union[int, str],
    ) -> str:
        if isinstance(scene_id, int):
            scene_id = f'{scene_id:04d}'
        if isinstance(frame_id, int):
            frame_id = f'{frame_id:06d}'
        return f'{scene_id:s}/{frame_id:s}'

    @classmethod
    def download(cls) -> None:
        url: str = 'https://drive.google.com/uc?id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi'  # NOQA
        md5 = None  # 'c9122e177a766a9691cab13c5cda41a9'
        gdown.cached_download(
            url=url,
            path=str(cls._root_dir) + '.zip',
            md5=md5,
            postprocess=gdown.extractall,
        )

    def get_ids(
        self,
        sampling: int = 1,
    ):
        split = self.split
        imageset_file: pathlib.Path = self.root_dir / f'image_sets/{split}.txt'
        with open(imageset_file) as f:
            ids = []
            for line in f:
                image_id = line.strip()
                video_id, frame_id = image_id.split('/')
                if (int(frame_id) - 1) % sampling == 0:
                    # frame_id starts from 1
                    ids.append(image_id)
            return tuple(ids)

    @classmethod
    def get_frame(cls, image_id: str) -> dict:
        meta_file: pathlib.Path = (
            cls._root_dir / cls._data_dir / (image_id + '-meta.mat')
        )
        meta = scipy.io.loadmat(
            meta_file, squeeze_me=True, struct_as_record=True
        )

        color_file: pathlib.Path = (
            cls._root_dir / cls._data_dir / (image_id + '-color.png')
        )
        color: np.ndarray = imgviz.io.imread(color_file)
        if color.shape[2] == 4:
            color = color[:, :, :3]  # rgba -> rgb

        depth_file: pathlib.Path = (
            cls._root_dir / cls._data_dir / (image_id + '-depth.png')
        )
        depth: np.ndarray = imgviz.io.imread(depth_file)
        depth = depth.astype(float) / meta['factor_depth']
        depth[depth == 0] = float('nan')

        label_file: pathlib.Path = (
            cls._root_dir / cls._data_dir / (image_id + '-label.png')
        )
        label: np.ndarray = imgviz.io.imread(label_file)

        return dict(
            meta=meta,
            color=color,
            depth=depth,
            label=label,
        )
