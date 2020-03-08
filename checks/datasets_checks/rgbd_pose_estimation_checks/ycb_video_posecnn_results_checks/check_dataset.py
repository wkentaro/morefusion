#!/usr/bin/env python

import sys

import path

import morefusion

here = path.Path(__file__).abspath().parent
sys.path.insert(0, here / "../ycb_video_checks")
from check_dataset import get_scene  # NOQA


if __name__ == "__main__":
    dataset = (
        morefusion.datasets.YCBVideoPoseCNNResultsRGBDPoseEstimationDataset()
    )  # NOQA
    morefusion.extra.trimesh.display_scenes(
        get_scene(dataset), height=int(320 * 0.7), width=int(480 * 0.7)
    )
