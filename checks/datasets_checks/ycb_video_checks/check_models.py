#!/usr/bin/env python

import concurrent.futures
import trimesh

import objslampp


def get_uniform_scale_cad(models, class_name):
    cad_file = models.get_cad_file(class_name=class_name)

    cad = trimesh.load(str(cad_file), file_type='obj', process=False)
    cad.visual = cad.visual.to_color()  # texture visualization is slow

    scale = cad.bounding_box.extents.max()
    cad.apply_scale(0.5 / scale)
    return cad


def main():
    models = objslampp.datasets.YCBVideoModels()
    class_names = objslampp.datasets.ycb_video.class_names

    cads = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for class_name in class_names[1:]:
            result = executor.submit(get_uniform_scale_cad, models, class_name)
            cads.append(result)
    cads = [future.result() for future in cads]

    scene = objslampp.extra.trimesh.tile_meshes(cads)
    scene.show()


if __name__ == '__main__':
    main()
