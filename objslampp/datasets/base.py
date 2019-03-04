import abc


class DatasetBase(abc.ABC):

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

    def __getitem__(self, i):
        id = self.ids[i]
        if isinstance(id, tuple):
            return self.getitem_from_id(*id)
        elif isinstance(id, dict):
            return self.getitem_from_id(**id)
        return self.getitem_from_id(id)

    @abc.abstractmethod
    def getitem_from_id(self, id):
        raise NotImplementedError
