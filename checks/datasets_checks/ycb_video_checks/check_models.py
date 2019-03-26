#!/usr/bin/env python

import tqdm
import trimesh

import objslampp


def main():
    models = objslampp.datasets.YCBVideoModels()
    class_names = objslampp.datasets.ycb_video.class_names

    cads = []
    min_z = float('inf')
    for class_name in tqdm.tqdm(class_names[1:]):
        model = models.get_model(class_name=class_name)
        cad_file = model['textured_simple']

        cad = trimesh.load(str(cad_file), file_type='obj', process=False)
        cad.visual = cad.visual.to_color()  # texture visualization is slow

        scale = cad.bounding_box.extents.max()
        cad.apply_scale(0.5 / scale)

        min_z = min(min_z, cad.vertices.min())

        cads.append(cad)

    scene = objslampp.extra.trimesh.tile_meshes(cads)
    scene.show()


if __name__ == '__main__':
    main()
