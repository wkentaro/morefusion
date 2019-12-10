from ..reindex import reindex
from .dataset import YCBVideoPoseCNNResultsRGBDPoseEstimationDataset


if __name__ == '__main__':
    dataset = YCBVideoPoseCNNResultsRGBDPoseEstimationDataset()
    reindexed_root_dir = dataset.root_dir + '.reindexed.v2'
    reindex(reindexed_root_dir, [dataset])
