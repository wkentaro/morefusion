#!/usr/bin/env python

import sys

import objslampp

from common import Inference

sys.path.insert(0, '../preliminary')  # NOQA
from align_occupancy_grids import refinement


models = objslampp.datasets.YCBVideoModels()
inference = Inference(gpu=0)
frame, Ts_cad2cam_true, Ts_cad2cam_pred = inference(index=0, bg_class=True)

keep = frame['class_ids'] > 0
class_ids_fg = frame['class_ids'][keep]
instance_ids_fg = frame['instance_ids'][keep]
Ts_cad2cam_true_fg = frame['Ts_cad2cam'][keep]

refinement(
    instance_ids=instance_ids_fg,
    class_ids=class_ids_fg,
    rgb=frame['rgb'],
    pcd=frame['pcd'],
    instance_label=frame['instance_label'],
    Ts_cad2cam_true=Ts_cad2cam_true,
    Ts_cad2cam_pred=Ts_cad2cam_pred,
)
