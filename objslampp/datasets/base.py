import pathlib
import typing

import chainer


class DatasetBase(chainer.dataset.DatasetMixin):

    _root_dir: typing.Optional[pathlib.Path] = None
    _split: typing.Optional[str] = None
    _ids: typing.Optional[list] = None

    @property
    def root_dir(self) -> pathlib.Path:
        if self._root_dir is None:
            raise ValueError('self._root_dir is not set')
        return self._root_dir

    @property
    def split(self) -> str:
        if self._split is None:
            raise ValueError('self._split is not set')
        return self._split

    @property
    def ids(self) -> list:
        if self._ids is None:
            raise ValueError('self._ids is not set')
        return self._ids

    def __len__(self) -> int:
        return len(self.ids)
