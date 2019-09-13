#!/usr/bin/env python

import numpy as np
import pybullet

import objslampp


def main():
    models = objslampp.datasets.YCBVideoModels()

    random_state = np.random.RandomState(0)
    generator = objslampp.simulation.BinTypeSceneGeneration(
        models=models, n_object=5, random_state=random_state
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-45,
        cameraTargetPosition=(0, 0, 0),
    )
    generator.generate()

    eye = objslampp.geometry.points_from_angles(
        distance=[1], elevation=[45], azimuth=[45],
    )[0]
    T_camera2world = objslampp.geometry.look_at(
        eye=eye,
        target=(0, 0, 0),
        up=(0, 0, -1),
    )
    generator.debug_render(T_camera2world)


if __name__ == '__main__':
    main()
