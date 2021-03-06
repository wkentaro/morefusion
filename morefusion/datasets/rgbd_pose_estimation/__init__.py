# flake8: noqa

from .my_real import MyRealRGBDPoseEstimationDataset

from .my_synthetic import MySyntheticRGBDPoseEstimationDataset

from .my_synthetic_ycb20190916 import (
    MySyntheticYCB20190916RGBDPoseEstimationDataset,
)
from .my_synthetic_ycb20190916 import (
    MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed,
)

from .ycb_video import YCBVideoRGBDPoseEstimationDataset
from .ycb_video import YCBVideoRGBDPoseEstimationDatasetReIndexed

from .ycb_video_posecnn_results import (
    YCBVideoPoseCNNResultsRGBDPoseEstimationDataset,
)
from .ycb_video_posecnn_results import (
    YCBVideoPoseCNNResultsRGBDPoseEstimationDatasetReIndexed,
)
