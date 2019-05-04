#!/usr/bin/env python

import sys

import objslampp

from common import Inference

sys.path.insert(0, '../preliminary')  # NOQA
from align_pointclouds import refinement


models = objslampp.datasets.YCBVideoModels()
inference = Inference(gpu=0)
frame, Ts_cad2cam_true, Ts_cad2cam_pred = inference(index=0)

K = frame['intrinsic_matrix']
pcd = objslampp.geometry.pointcloud_from_depth(
    frame['depth'], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
)

refinement(
    instance_ids=frame['instance_ids'],
    class_ids=frame['class_ids'],
    rgb=frame['rgb'],
    pcd=pcd,
    instance_label=frame['instance_label'],
    Ts_cad2cam_true=frame['Ts_cad2cam'],
    Ts_cad2cam_pred=Ts_cad2cam_pred,
)
