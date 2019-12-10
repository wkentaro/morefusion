import shlex
import subprocess

import chainer
import gdown
import scipy.io

from ..base import DatasetBase
from .dataset import YCBVideoDataset


class YCBVideoPoseCNNResultsDataset(DatasetBase):

    _root_dir = chainer.dataset.get_dataset_directory(
        'ycb_video/YCB_Video_toolbox/results_PoseCNN_RSS2018',
        create_directory=False,
    )

    def __init__(self):
        self.download()
        self._parent = YCBVideoDataset(split='keyframe', sampling=1)
        self._ids = self._parent._ids

    def download(self) -> None:
        if not self.root_dir.exists():
            url = 'https://github.com/yuxng/YCB_Video_toolbox.git'
            cmd = f'git clone {url} {self.root_dir}'
            subprocess.check_call(shlex.split(cmd))
            gdown.extractall(self.root_dir / 'results_PoseCNN_RSS2018.zip')

    def get_example(self, i):
        image_id = self._ids[i]
        example = self._parent.get_frame(image_id)
        result_file = self.root_dir / f'{i:06d}.mat'
        result = scipy.io.loadmat(
            result_file, squeeze_me=True, struct_as_record=True
        )
        example['result'] = result
        return example
