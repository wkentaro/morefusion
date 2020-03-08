#!/usr/bin/env python

import time

import chainer
from chainer import cuda
import numpy as np

import morefusion


models = morefusion.datasets.YCBVideoModels()

points_list = []

iteration = 21
print(f"iteration: {iteration}")

for class_id in range(1, 1 + iteration):
    points = models.get_pcd(class_id=class_id)
    points = morefusion.extra.open3d.voxel_down_sample(points, voxel_size=0.01)
    points = points.astype(np.float32)
    points = chainer.cuda.to_gpu(points)
    points_list.append(points)

dim = 16
pitch = 0.01
origin = np.array((-pitch * dim / 2,) * 3, dtype=np.float32)
dims = (dim,) * 3
print(f"pitch: {pitch}")
print(f"dim: {dim}")

for func in ["legacy", "latest"]:
    if func == "legacy":
        function = morefusion.functions.occupancy_grid_3d
    else:
        assert func == "latest"
        function = morefusion.functions.pseudo_occupancy_voxelization

    cuda.Stream().synchronize()
    t_start = time.time()
    for points in points_list:
        grid = function(
            points, pitch=pitch, origin=origin, dims=dims, threshold=2,
        )
    cuda.Stream().synchronize()
    print(f"[{func}] {time.time() - t_start} [s]")
