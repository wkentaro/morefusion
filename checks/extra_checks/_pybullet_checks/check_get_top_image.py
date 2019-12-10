#!/usr/bin/env python

import concurrent.futures

import imgviz

import morefusion


def _get_top_image(class_id):
    models = morefusion.datasets.YCBVideoModels()
    cad_file = models.get_cad_file(class_id=class_id)
    return morefusion.extra.pybullet.get_top_image(cad_file)


def main():
    models = morefusion.datasets.YCBVideoModels()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for class_id in range(models.n_class):
            if class_id == 0:
                continue
            future = executor.submit(_get_top_image, class_id)
            futures.append(future)

    viz = []
    for future in futures:
        viz_i = future.result()
        viz.append(viz_i)
    viz = imgviz.tile(viz, shape=(4, 6))
    imgviz.io.pyglet_imshow(viz)
    imgviz.io.pyglet_run()


if __name__ == '__main__':
    main()
