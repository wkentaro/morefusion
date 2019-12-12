#!/usr/bin/env python

import numpy as np
import trimesh

import morefusion


thickness = 0.01
xlength = 0.3
ylength = 0.5
zlength = 0.2

mesh = morefusion.extra.trimesh.bin_model(
    (xlength, ylength, zlength), thickness
)

scene = trimesh.Scene()
scene.add_geometry(trimesh.creation.axis(0.01))
scene.add_geometry(mesh)
morefusion.extra.trimesh.display_scenes(
    {'scene': scene},
    rotate=True,
)
