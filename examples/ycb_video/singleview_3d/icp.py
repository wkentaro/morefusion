#!/usr/bin/env python

import sys

import numpy as np

import morefusion

from common import Inference

sys.path.insert(0, '../preliminary')  # NOQA
from align_pointclouds import refinement


models = morefusion.datasets.YCBVideoModels()
inference = Inference(dataset='my_real', gpu=0)
frame, Ts_cad2cam_true, Ts_cad2cam_pred = inference(index=0)

keep = np.isin(frame['class_ids'], inference.dataset._class_ids)
frame['class_ids'] = frame['class_ids'][keep]
frame['instance_ids'] = frame['instance_ids'][keep]
frame['Ts_cad2cam'] = frame['Ts_cad2cam'][keep]

refinement(
    instance_ids=frame['instance_ids'],
    class_ids=frame['class_ids'],
    rgb=frame['rgb'],
    pcd=frame['pcd'],
    instance_label=frame['instance_label'],
    Ts_cad2cam_true=frame['Ts_cad2cam'],
    Ts_cad2cam_pred=Ts_cad2cam_pred,
)
