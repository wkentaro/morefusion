#!/usr/bin/env python

import argparse
import sys

import numpy as np
import trimesh

import objslampp

from common import Inference

sys.path.insert(0, '../preliminary')  # NOQA
from align_occupancy_grids import refinement


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--prior', action='store_true', help='prior of ground and bin'
)
args = parser.parse_args()

models = objslampp.datasets.YCBVideoModels()
inference = Inference(gpu=0)
frame, Ts_cad2cam_true, Ts_cad2cam_pred = inference(index=0, bg_class=True)

points_occupied = {}
if args.prior:
    # ground
    dim = (64, 64, 16)
    pitch = 0.01
    matrix = np.ones(dim, dtype=bool)
    origin = - dim[0] * pitch / 2, - dim[1] * pitch / 2, - dim[2] * pitch
    points = trimesh.voxel.matrix_to_points(matrix, pitch, origin)
    points = trimesh.transform_points(
        points, frame['Ts_cad2cam'][frame['instance_ids'] == 0][0]
    )
    points_occupied[0] = points
    # bin
    mesh = trimesh.load(str(frame['cad_files'][1]))
    mesh.apply_transform(frame['Ts_cad2cam'][frame['instance_ids'] == 1][0])
    points_occupied[1] = mesh.voxelized(0.01).points

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
    points_occupied=points_occupied,
)
