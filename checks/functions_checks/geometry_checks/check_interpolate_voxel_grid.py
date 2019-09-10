#!/usr/bin/env python

from chainer.backends import cuda
import numpy as np
import trimesh
import trimesh.viewer

import objslampp


dataset = objslampp.datasets.YCBVideoRGBDPoseEstimationDataset('train')
example = dataset[0][0]

pitch = example['pitch']
origin = example['origin']
dim = 32

mapping = objslampp.geometry.VoxelMapping(
    origin=origin,
    pitch=pitch,
    voxel_dim=dim,
    nchannel=3,
)

mask = ~np.isnan(example['pcd']).any(axis=2)
points = example['pcd'][mask]
values = example['rgb'][mask]
mapping.add(points, values)

indices = (points - origin) / pitch
I, J, K = indices[:, 0], indices[:, 1], indices[:, 2]
keep = (0 <= I) & (I < dim) & (0 <= J) & (J < dim) & (0 <= K) & (K < dim)
I, J, K = I[keep], J[keep], K[keep]
points_extracted = points[keep]
indices_extracted = indices[keep]

scenes = {}

geom = trimesh.PointCloud(points_extracted, values[keep])
scenes['original'] = trimesh.Scene(geom)

I = I.round().astype(int)
J = J.round().astype(int)
K = K.round().astype(int)
values_extracted = mapping.values[I, J, K] / 255.
geom = trimesh.PointCloud(points_extracted, colors=values_extracted)
scenes['integer_indexing'] = trimesh.Scene(
    geom, camera=scenes['original'].camera
)

voxelized = mapping.values.astype(np.float32).transpose(3, 0, 1, 2)
values_extracted = objslampp.functions.interpolate_voxel_grid(
    cuda.to_gpu(voxelized / 255.),
    cuda.to_gpu(indices_extracted.astype(np.float32)),
)
values_extracted = cuda.to_cpu(values_extracted.array)
geom = trimesh.PointCloud(points_extracted, colors=values_extracted)
scenes['float_indexing'] = trimesh.Scene(
    geom, camera=scenes['original'].camera
)

objslampp.extra.trimesh.display_scenes(
    scenes,
    tile=(1, 3),
    height=int(480 * 0.8),
    width=int(640 * 0.8),
)
