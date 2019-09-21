#!/usr/bin/env python

import sys

import path

import objslampp

here = path.Path(__file__).abspath().parent
sys.path.insert(0, here / '../ycb_video_checks')
from check_dataset import get_scene  # NOQA


if __name__ == '__main__':
    dataset = objslampp.datasets.MySyntheticYCB20190916RGBDPoseEstimationDataset('train')  # NOQA
    objslampp.extra.trimesh.display_scenes(
        get_scene(dataset), height=int(320 * 0.5), width=int(480 * 0.5)
    )
