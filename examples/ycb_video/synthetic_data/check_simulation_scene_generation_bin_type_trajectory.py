#!/usr/bin/env python

import imgviz
import numpy as np
import pybullet
import termcolor

import objslampp

import contrib


def main():
    models = objslampp.datasets.YCBVideoModels()

    random_state = np.random.RandomState(0)
    generator = contrib.simulation.BinTypeSceneGeneration(
        extents=(0.3, 0.5, 0.3),
        models=models,
        n_object=10,
        random_state=random_state,
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )
    generator.generate()

    self = generator

    n_keypoints = 10
    n_points = 72

    # targets
    targets = self._random_state.uniform(*self._aabb, (n_keypoints, 3))
    targets = contrib.geometry.trajectory.sort(targets)
    targets = contrib.geometry.trajectory.interpolate(
        targets, n_points=n_points
    )
    for i, target in enumerate(targets):
        termcolor.cprint(f'==> target[{i}]: {target}', attrs={'bold': True})

    # eyes
    distance = np.full((n_keypoints,), 1, float)
    elevation = self._random_state.uniform(30, 90, (n_keypoints,))
    azimuth = self._random_state.uniform(0, 360, (n_keypoints,))
    eyes = objslampp.geometry.points_from_angles(distance, elevation, azimuth)
    indices = np.linspace(0, n_points - 1, num=len(eyes))
    indices = indices.round().astype(int)
    eyes = contrib.geometry.trajectory.sort_by(eyes, key=targets[indices])
    eyes = contrib.geometry.trajectory.interpolate(eyes, n_points=n_points)
    for i, eye in enumerate(eyes):
        termcolor.cprint(f'==> eye[{i}]: {eye}', attrs={'bold': True})

    # visualize
    def images(self, eyes, targets):
        depth2rgb = imgviz.Depth2RGB()
        n_points = len(eyes)
        for i in range(n_points):
            T_camera2world = objslampp.geometry.look_at(
                eye=eyes[i], at=targets[i]
            )
            # generator.debug_render(T_camera2world)

            rgb, depth, ins, cls = self.render(
                T_camera2world, fovy=45, height=480, width=640,
            )
            viz = imgviz.tile([
                rgb,
                depth2rgb(depth),
                imgviz.label2rgb(ins + 1, rgb),
                imgviz.label2rgb(cls, rgb),
            ], border=(255, 255, 255))
            viz = imgviz.resize(viz, width=1000)

            font_size = 25
            text = f'{i + 1:04d} / {n_points:04d}'
            size = imgviz.draw.text_size(text, font_size)
            viz = imgviz.draw.rectangle(
                viz, (1, 1), size, color=(0, 255, 0), fill=(0, 255, 0)
            )
            viz = imgviz.draw.text(
                viz, (1, 1), text, color=(0, 0, 0), size=font_size
            )

            yield viz

    imgviz.io.pyglet_imshow(images(self, eyes, targets))
    imgviz.io.pyglet_run()


if __name__ == '__main__':
    main()
