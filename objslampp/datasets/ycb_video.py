import pathlib
import typing

import gdown
import imgviz
import numpy as np
import scipy.io


class YCBVideoDataset(object):

    root_dir: pathlib.Path = \
        pathlib.Path.home() / 'data/datasets/YCB/YCB_Video_Dataset'

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
            path=str(cls.root_dir) + '.zip',
            md5=md5,
            postprocess=gdown.extractall,
        )

    @classmethod
    def get_imageset(cls, split: str) -> typing.Tuple[str, ...]:
        assert split in ['train', 'val', 'trainval']
        imageset_file: pathlib.Path = cls.root_dir / f'image_sets/{split}.txt'
        with open(imageset_file) as f:
            imageset: typing.Tuple[str, ...] = tuple(
                [l.strip() for l in f.readlines()]
            )
        return imageset

    @classmethod
    def get_frame(cls, image_id: str) -> dict:
        meta_file: pathlib.Path = (
            cls.root_dir / 'data' / (image_id + '-meta.mat')
        )
        meta = scipy.io.loadmat(
            meta_file, squeeze_me=True, struct_as_record=True
        )

        color_file: pathlib.Path = (
            cls.root_dir / 'data' / (image_id + '-color.png')
        )
        color: np.ndarray = imgviz.io.imread(color_file)

        depth_file: pathlib.Path = (
            cls.root_dir / 'data' / (image_id + '-depth.png')
        )
        depth: np.ndarray = imgviz.io.imread(depth_file)
        depth = depth.astype(float) / meta['factor_depth']
        depth[depth == 0] = float('nan')

        label_file: pathlib.Path = (
            cls.root_dir / 'data' / (image_id + '-label.png')
        )
        label: np.ndarray = imgviz.io.imread(label_file)

        return dict(
            meta=meta,
            color=color,
            depth=depth,
            label=label,
        )

    def __init__(self, split: str):
        assert split in ['train', 'val', 'trainval']
        self._split = split
        self._imageset = self.get_imageset(split)

        if not self.root_dir.exists():
            self.download()

    @property
    def split(self):
        return self._split

    @property
    def imageset(self):
        return self._imageset

    def __len__(self):
        return len(self.imageset)

    def __getitem__(self, i):
        image_id = self.imageset[i]
        return self.get_frame(image_id)


class_names = (
    '__background__',
    '002_master_chef_can',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '009_gelatin_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '036_wood_block',
    '037_scissors',
    '040_large_marker',
    '051_large_clamp',
    '052_extra_large_clamp',
    '061_foam_brick',
)
