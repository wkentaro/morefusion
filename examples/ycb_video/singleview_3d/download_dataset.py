#!/usr/bin/env python

import morefusion


# download() is called in __init__()
morefusion.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed(
    split="trainreal"
)
morefusion.datasets.MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed(
    split="train"
)
