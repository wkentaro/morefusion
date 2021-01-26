#!/usr/bin/env python

import open3d

import morefusion


models = morefusion.datasets.YCBVideoModels()
cad = models.get_cad(class_id=1)
cad.visual = cad.visual.to_color()

cad = morefusion.utils.trimesh_to_open3d(cad)
open3d.visualization.draw_geometries([cad])
