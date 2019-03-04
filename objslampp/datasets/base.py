import chainer


class DatasetBase(chainer.dataset.DatasetMixin):

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def split(self):
        return self._split

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)
