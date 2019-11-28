#!/usr/bin/env python

import concurrent.futures

import objslampp


def get_uniform_scale_cad(models, class_id):
    cad = models.get_cad(class_id=class_id)
    cad.visual = cad.visual.to_color()  # texture visualization is slow

    scale = cad.bounding_box.extents.max()
    cad.apply_scale(0.5 / scale)
    return cad


def main():
    models = objslampp.datasets.YCBVideoModels()
    class_names = objslampp.datasets.ycb_video.class_names

    cads = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for class_id, class_name in enumerate(class_names):
            if class_id == 0:
                continue
            result = executor.submit(get_uniform_scale_cad, models, class_id)
            cads.append(result)
    cads = [future.result() for future in cads]

    scene = objslampp.extra.trimesh.tile_meshes(cads)
    scene.show()


if __name__ == '__main__':
    main()
