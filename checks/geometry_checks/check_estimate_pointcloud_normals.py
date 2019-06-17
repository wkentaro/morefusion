#!/usr/bin/env python

import imgviz
import numpy as np

import objslampp


def main():
    example = objslampp.datasets.YCBVideoDataset('train')[0]
    depth = example['depth']
    K = example['meta']['intrinsic_matrix']
    pcd = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

    normals_organized = objslampp.geometry.estimate_pointcloud_normals(pcd)

    nonnan = ~np.isnan(depth)
    normals_unorganized = np.full_like(pcd, -1)
    normals_unorganized[nonnan] = \
        objslampp.geometry.estimate_pointcloud_normals(pcd[nonnan])

    normals_organized = np.uint8((normals_organized + 1) * 128)
    normals_unorganized = np.uint8((normals_unorganized + 1) * 128)

    viz = imgviz.tile([normals_organized, normals_unorganized],
                      (1, 2), border=(255, 255, 255))
    imgviz.io.pyglet_imshow(viz)
    imgviz.io.pyglet_run()


if __name__ == '__main__':
    main()
