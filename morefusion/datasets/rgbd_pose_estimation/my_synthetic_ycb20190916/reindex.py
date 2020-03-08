from ..reindex import reindex
from .dataset import MySyntheticYCB20190916RGBDPoseEstimationDataset


if __name__ == "__main__":
    datasets = [
        MySyntheticYCB20190916RGBDPoseEstimationDataset("train"),
        MySyntheticYCB20190916RGBDPoseEstimationDataset("val"),
    ]
    reindexed_root_dir = datasets[0].root_dir + ".reindexed"
    reindex(reindexed_root_dir, datasets)
