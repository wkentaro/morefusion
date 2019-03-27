from .dataset import YCBVideoDataset


class YCBVideoSyntheticDataset(YCBVideoDataset):

    _data_dir = 'data_syn'

    def __init__(self):
        self._ids = self.get_ids()

    def get_ids(self):
        data_dir = self.root_dir / 'data_syn'
        ids = sorted(x.name.split('-')[0] for x in data_dir.glob('*-meta.mat'))
        return ids


if __name__ == '__main__':
    dataset = YCBVideoSyntheticDataset()
