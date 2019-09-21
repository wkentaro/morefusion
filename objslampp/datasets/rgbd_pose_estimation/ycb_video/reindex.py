from ..reindex import reindex
from .dataset import YCBVideoRGBDPoseEstimationDataset


if __name__ == '__main__':
    datasets = [
        YCBVideoRGBDPoseEstimationDataset('val'),
        YCBVideoRGBDPoseEstimationDataset('train'),
    ]
    reindexed_root_dir = datasets[0].root_dir + '.reindexed.test'
    reindex(reindexed_root_dir, datasets)
