from .dataset import YCBVideoDataset


class YCBVideoSyntheticDataset(YCBVideoDataset):

    _data_dir = "data_syn"

    def __init__(self):
        self._split = "syn"
        self._ids = self.get_ids()
        self._id_to_class_ids = None

        if not self.root_dir.exists():
            self.download()

    def get_ids(self):
        data_dir = self.root_dir / "data_syn"
        ids = sorted(x.name.split("-")[0] for x in data_dir.glob("*-meta.mat"))
        return ids


if __name__ == "__main__":
    dataset = YCBVideoSyntheticDataset()
