#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas
import trimesh

import objslampp


def main():
    models = objslampp.datasets.YCBVideoModels()

    data = []
    for model_dir in sorted(models.root_dir.iterdir()):
        cad_file = model_dir / f'textured_simple.obj'
        cad = trimesh.load(str(cad_file), file_type='obj', process=False)
        cad.visual = cad.visual.to_color()  # texture visualization is slow

        extents = cad.bounding_box.extents
        bbox_max_size = np.sqrt((extents ** 2).sum())
        voxel_size = bbox_max_size / 32.

        data.append((model_dir.name, bbox_max_size, voxel_size))

    df = pandas.DataFrame(
        data=data, columns=['name', 'bbox_max_size', 'voxel_size']
    )
    print(df)

    df = df.sort_values('voxel_size')[::-1]
    df.plot.bar(
        x='name',
        y='voxel_size',
        color=(0.1, 0.1, 0.1, 0.1),
        edgecolor='red',
        rot=45,
    )
    plt.title('Voxel size of YCB_Video_Models')
    plt.show()


if __name__ == '__main__':
    main()
