#!/usr/bin/env python

import imgviz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import numpy as np
import pandas
import trimesh

import objslampp

from tmp import ResNetFeatureExtractor


def get_uniform_points_on_sphere(radius=1, n_sample=10):
    elevation = np.linspace(-90, 90, n_sample)
    azimuth = np.linspace(-180, 180, n_sample, endpoint=False)
    elevation, azimuth = np.meshgrid(elevation, azimuth)

    # if elevation is -90 or 90, azimuth has no effect
    keep = elevation != -90
    keep[np.argmin(keep)] = True
    azimuth = azimuth[keep]
    elevation = elevation[keep]

    keep = elevation != 90
    keep[np.argmin(keep)] = True
    azimuth = azimuth[keep]
    elevation = elevation[keep]

    elevation = elevation.flatten()
    azimuth = azimuth.flatten()

    n_points = len(elevation)
    distance = np.full((n_points,), radius, dtype=float)
    points = objslampp.geometry.get_points_from_angles(
        distance, elevation, azimuth
    )
    return points


def get_rendered(visual_file, eyes, targets, height=256, width=256, gui=False):
    import pybullet

    if gui:
        pybullet.connect(pybullet.GUI)
    else:
        pybullet.connect(pybullet.DIRECT)

    objslampp.sim.pybullet.add_model(visual_file=visual_file, register=False)

    near = 0.01
    far = 1000.
    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=60, aspect=1. * width / height, nearVal=near, farVal=far
    )

    rendered = []
    for eye, target in zip(eyes, targets):
        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=[0, 1, 0],
        )
        H, W, rgba, depth, segm = pybullet.getCameraImage(
            height,
            width,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
        )
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(H, W, 4)
        rgb = rgba[:, :, :3]

        segm = np.asarray(segm, dtype=np.int32)

        depth = np.asarray(depth, dtype=np.float32).reshape(H, W)
        depth = far * near / (far - (far - near) * depth)
        depth[segm == -1] = np.nan

        rendered.append((rgb, depth, segm))

    pybullet.disconnect()

    return rendered


class MainApp(object):

    def _get_eyes(self):
        eyes = get_uniform_points_on_sphere(radius=0.3, n_sample=5)
        return eyes

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

    def plot_views(self, res=False, gpu=0):
        if res:
            res = ResNetFeatureExtractor(gpu=gpu)

        models = objslampp.datasets.YCBVideoModels()
        visual_file = (
            models.root_dir / '002_master_chef_can/textured_simple.obj'
        )

        eyes = self._get_eyes()
        targets = np.tile([[0, 0, 0]], (len(eyes), 1))
        rendered = get_rendered(visual_file, eyes, targets)

        viz = []
        depth2rgb = imgviz.Depth2RGB()
        for rgb, depth, segm in rendered:
            depth_viz = depth2rgb(depth)
            mask = (segm == 0).astype(np.uint8) * 255
            if res:
                feat = res.extract_feature(rgb)
                feat_viz = res.feature2rgb(feat, segm != -1)
                viz.append(
                    imgviz.tile([rgb, depth_viz, mask, feat_viz], (1, 4))
                )
            else:
                viz.append(imgviz.tile([rgb, depth_viz, mask]))

        viz = imgviz.tile(viz, border=(127, 127, 127))
        viz = imgviz.resize(viz, height=500)
        imgviz.io.cv_imshow(viz, __file__)
        while imgviz.io.cv_waitkey() != ord('q'):
            pass

    def plot_pointcloud(self):
        models = objslampp.datasets.YCBVideoModels()
        visual_file = (
            models.root_dir / '002_master_chef_can/textured_simple.obj'
        )

        eyes = self._get_eyes()
        targets = np.tile([[0, 0, 0]], (len(eyes), 1))
        rendered = get_rendered(visual_file, eyes, targets)

        # ---------------------------------------------------------------------

        scene = trimesh.Scene()

        # world origin
        geom = trimesh.creation.axis(
            origin_size=0.01, origin_color=(255, 0, 0)
        )
        scene.add_geometry(geom)

        camera = trimesh.scene.Camera(resolution=(256, 256), fov=(60, 60))
        K = camera.K
        for eye, target, (rgb, depth, segm) in zip(eyes, targets, rendered):

            # it transforms: camera frame -> world frame
            T = objslampp.geometry.look_at(eye, target, up=[0, -1, 0])

            # camera origin
            geom = trimesh.creation.axis(origin_size=0.01)
            geom.apply_transform(T)
            scene.add_geometry(geom)

            points = objslampp.geometry.pointcloud_from_depth(
                depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
            )

            valid = ~np.isnan(depth)
            colors = rgb[valid]
            points = points[valid]

            points = trimesh.transform_points(points, T)

            geom = trimesh.PointCloud(vertices=points, color=colors)
            scene.add_geometry(geom)

        scene.show()

    def voxel_mapping(self, channel='rgb', gpu=0):
        assert channel in ['rgb', 'res']

        if channel == 'res':
            res = ResNetFeatureExtractor(gpu=gpu)

        models = objslampp.datasets.YCBVideoModels()
        visual_file = (
            models.root_dir / '002_master_chef_can/textured_simple.obj'
        )

        eyes = self._get_eyes()
        targets = np.tile([[0, 0, 0]], (len(eyes), 1))
        rendered = get_rendered(visual_file, eyes, targets)

        # ---------------------------------------------------------------------

        class_id = 2
        df = pandas.read_csv('data/voxel_size.csv')
        pitch = float(df['voxel_size'][df['class_id'] == class_id])

        mapping = objslampp.geometry.VoxelMapping(
            origin=(-16 * pitch,) * 3,
            pitch=pitch,
            voxel_size=32,
            nchannel=3,
        )

        camera = trimesh.scene.Camera(resolution=(256, 256), fov=(60, 60))
        K = camera.K
        for eye, target, (rgb, depth, segm) in zip(eyes, targets, rendered):
            if channel == 'res':
                feat = res.extract_feature(rgb)
                rgb = res.feature2rgb(feat, segm != -1)

            # it transforms: camera frame -> world frame
            T = objslampp.geometry.look_at(eye, target, up=[0, -1, 0])

            points = objslampp.geometry.pointcloud_from_depth(
                depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
            )

            valid = ~np.isnan(depth)
            colors = rgb[valid]
            points = points[valid]

            points = trimesh.transform_points(points, T)

            mapping.add(points=points, values=colors / 255.)

        scene = trimesh.Scene()

        scene.add_geometry(trimesh.creation.axis(0.01))

        geom = trimesh.creation.axis(0.01)
        geom.apply_translation(mapping.origin)
        scene.add_geometry(geom)

        geom = mapping.as_boxes()
        scene.add_geometry(geom)

        geom = mapping.as_bbox()
        scene.add_geometry(geom)

        scene.show()


if __name__ == '__main__':
    import fire

    fire.Fire(MainApp)
