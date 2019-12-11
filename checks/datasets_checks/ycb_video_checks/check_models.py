#!/usr/bin/env python

import concurrent.futures

import morefusion


def get_uniform_scale_cad(models, class_id):
    cad = models.get_cad(class_id=class_id)
    cad.visual = cad.visual.to_color()  # texture visualization is slow

    scale = cad.bounding_box.extents.max()
    cad.apply_scale(0.5 / scale)
    return cad


def main():
    models = morefusion.datasets.YCBVideoModels()
    class_names = morefusion.datasets.ycb_video.class_names

    cads = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for class_id, class_name in enumerate(class_names):
            if class_id == 0:
                continue
            result = executor.submit(get_uniform_scale_cad, models, class_id)
            cads[f'{class_id:04d}'] = result

    scenes = {}
    camera_transform = None
    for k, v in cads.items():
        cad = v.result()
        scene = cad.scene()
        if camera_transform is None:
            camera_transform = scene.camera_transform
        else:
            scene.camera_transform = camera_transform
        scenes[k] = scene

    morefusion.extra.trimesh.display_scenes(
        scenes, height=480 // 3, width=640 // 3, tile=(3, 7)
    )


if __name__ == '__main__':
    main()
