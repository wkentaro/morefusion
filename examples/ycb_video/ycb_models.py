#!/usr/bin/env python

import trimesh

import objslampp


def main():
    models = objslampp.datasets.YCBVideoModels()

    cads = []
    min_z = float('inf')
    for model_dir in sorted(models.root_dir.iterdir()):
        cad_file = model_dir / f'textured_simple.obj'
        cad = trimesh.load(str(cad_file), file_type='obj', process=False)
        cad.visual = cad.visual.to_color()  # texture visualization is slow

        scale = cad.bounding_box.extents.max()
        cad.apply_scale(0.5 / scale)

        min_z = min(min_z, cad.vertices.min())

        cads.append(cad)

    scene = objslampp.vis.trimesh.tile_meshes(cads)
    scene.show()


if __name__ == '__main__':
    main()
