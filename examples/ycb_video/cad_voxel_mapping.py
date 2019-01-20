#!/usr/bin/env python

import imgviz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import numpy as np
import pybullet
import trimesh

import objslampp


def get_uniform_points_on_sphere(radius=1, n_points=10 * 10):
    n_points_sqrt = int(np.sqrt(n_points).round())
    elevation = np.linspace(-90, 90, n_points_sqrt)
    azimuth = np.linspace(-180, 180, n_points_sqrt)
    elevation, azimuth = np.meshgrid(elevation, azimuth)
    elevation = elevation.flatten()
    azimuth = azimuth.flatten()
    distance = np.full((n_points,), radius, dtype=float)
    points = objslampp.geometry.get_points_from_angles(
        distance, elevation, azimuth
    )
    return points


def get_rendered(visual_file, eyes, targets, height=256, width=256):
    pybullet.connect(pybullet.DIRECT)

    objslampp.sim.pybullet.add_model(visual_file=visual_file, register=False)

    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=60, aspect=1. * width / height, nearVal=0.01, farVal=100,
    )

    rendered = []
    for eye, target in zip(eyes, targets):
        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=[0, -1, 0],
        )
        H, W, rgba, *_ = pybullet.getCameraImage(
            height,
            width,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
        )
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(H, W, 4)
        rgb = rgba[:, :, :3]
        mask = rgba[:, :, 3] != 0
        rendered.append((rgb, mask))

    pybullet.disconnect()

    return rendered


class MainApp(object):

    def _get_eyes(self):
        return get_uniform_points_on_sphere(radius=0.3, n_points=10 * 10)

    def plot_eyes(self):
        eyes = self._get_eyes()
        x, y, z = zip(*eyes)
        ax = plt.subplot(projection='3d')
        ax.scatter(x, y, z)
        plt.show()

    def plot_rays(self):
        eyes = self._get_eyes()
        targets = np.tile([[0, 0, 0]], (len(eyes), 1))

        scene = trimesh.Scene()

        geom = trimesh.PointCloud(vertices=eyes)
        scene.add_geometry(geom)

        for eye, target in zip(eyes, targets):
            geom = trimesh.load_path([eye, target])
            scene.add_geometry(geom)

        scene.show()

    def plot_views(self):
        models = objslampp.datasets.YCBVideoModels()
        visual_file = (
            models.root_dir / '002_master_chef_can/textured_simple.obj'
        )

        eyes = self._get_eyes()
        targets = np.tile([[0, 0, 0]], (len(eyes), 1))
        rendered = get_rendered(visual_file, eyes, targets)

        rgbs = list(zip(*rendered))[0]
        viz = imgviz.tile(rgbs, shape=(5, 20))
        viz = imgviz.resize(viz, height=500)
        imgviz.io.cv_imshow(viz)
        while imgviz.io.cv_waitkey() != ord('q'):
            pass


if __name__ == '__main__':
    import fire

    fire.Fire(MainApp)
