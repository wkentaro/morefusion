#!/usr/bin/env python

import trimesh

import morefusion


models = morefusion.datasets.YCBVideoModels()

vg = models.get_solid_voxel_grid(class_id=1)

scene = trimesh.Scene()
scene.add_geometry(vg.as_boxes(colors=(0, 127, 0)))
scene.add_geometry(morefusion.extra.trimesh.box_outline_from_voxel_grid(vg))
scene.add_geometry(trimesh.creation.axis(origin_size=0.01))
morefusion.extra.trimesh.display_scenes({"scene": scene})
