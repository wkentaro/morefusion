#!/usr/bin/env python

import time

import chainer
from chainer.backends import cuda
import numpy as np
import trimesh

import objslampp


def get_scenes():
    instances = []
    for instance_id in range(3):
        instances.append(np.load(f'{instance_id:08d}.npz'))

    models = objslampp.datasets.YCBVideoModels()

    transform = []
    points = []
    sdf = []
    pitch = []
    origin = []
    grid_target = []
    grid_nontarget_empty = []
    for instance in instances:
        transform.append(instance['transform_init'].astype(np.float32))
        points_i, sdf_i = models.get_sdf(class_id=instance['class_id'])
        points.append(cuda.to_gpu(points_i).astype(np.float32))
        sdf.append(cuda.to_gpu(sdf_i).astype(np.float32))
        pitch.append(instance['pitch'].astype(np.float32))
        origin.append(instance['origin'].astype(np.float32))
        grid_target.append(instance['grid_target'].astype(np.float32))
        grid_nontarget_empty.append(
            instance['grid_nontarget_empty'].astype(np.float32)
        )
    pitch = cuda.cupy.asarray(pitch)
    origin = cuda.cupy.asarray(origin)
    grid_target = cuda.cupy.asarray(grid_target)
    grid_nontarget_empty = cuda.cupy.asarray(grid_nontarget_empty)

    link = objslampp.contrib.CollisionBasedPoseRefinementLink(transform)
    link.to_gpu()
    optimizer = chainer.optimizers.Adam(alpha=0.01)
    optimizer.setup(link)
    link.translation.update_rule.hyperparam.alpha *= 0.1

    t_start = time.time()
    for i in range(200):
        transform = objslampp.functions.transformation_matrix(
            link.quaternion, link.translation
        )
        transform = cuda.to_cpu(transform.array)
        scenes = {'scene': trimesh.Scene()}
        for j, instance in enumerate(instances):
            cad = models.get_cad(instance['class_id'])
            if hasattr(cad.visual, 'to_color'):
                cad.visual = cad.visual.to_color()
            scenes['scene'].add_geometry(
                cad,
                node_name=str(j),
                geom_name=str(instance['class_id']),
                transform=transform[j],
            )
        scenes['scene'].camera.transform = \
            objslampp.extra.trimesh.to_opengl_transform()
        yield scenes

        loss = link(
            points, sdf, pitch, origin, grid_target, grid_nontarget_empty
        )
        loss.backward()
        optimizer.update()
        link.zerograds()

        print(i, time.time() - t_start)
        # print(i)
        # print(link.quaternion, link.quaternion.dtype)
        # print(link.translation, link.translation.dtype)


def main():
    scenes = get_scenes()
    objslampp.extra.trimesh.display_scenes(scenes)


if __name__ == '__main__':
    main()
