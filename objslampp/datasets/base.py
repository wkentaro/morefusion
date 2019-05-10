import typing

import chainer
import path


class DatasetBase(chainer.dataset.DatasetMixin):

    _root_dir: typing.Optional[str] = None
    _split: typing.Optional[str] = None
    _ids: typing.Optional[list] = None

    @property
    def root_dir(self) -> path.Path:
        if self._root_dir is None:
            raise ValueError('self._root_dir is not set')
        if type(self._root_dir) is not path.Path:
            self._root_dir = path.Path(self._root_dir)
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
