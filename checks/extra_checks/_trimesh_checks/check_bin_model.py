#!/usr/bin/env python

import numpy as np
import trimesh

import objslampp


thickness = 0.01
xlength = 0.3
ylength = 0.5
zlength = 0.2

mesh = objslampp.extra.trimesh.bin_model(
    (xlength, ylength, zlength), thickness
)

scene = trimesh.Scene()
scene.add_geometry(trimesh.creation.axis(0.01))
scene.add_geometry(mesh)
objslampp.extra.trimesh.show_with_rotation(
    scene,
    step=(0, 0, np.deg2rad(1)),
    init_angles=(np.deg2rad(45), 0, 0),
    resolution=(400, 400),
)
