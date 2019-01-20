#!/usr/bin/env python

import pathlib

import imgviz
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pybullet  # NOQA
import tqdm
import trimesh

import objslampp


here = pathlib.Path(__file__).resolve().parent


def main():
    models = objslampp.datasets.YCBVideoModels()

    top_images = []
    data = []
    for model_dir in tqdm.tqdm(sorted(models.root_dir.iterdir())):
        cad_file = model_dir / f'textured_simple.obj'

        top_image = objslampp.sim.pybullet.get_top_image(cad_file)
        top_images.append(top_image)

        cad = trimesh.load(str(cad_file), file_type='obj', process=False)

        extents = cad.bounding_box.extents
        bbox_max_size = np.sqrt((extents ** 2).sum())
        voxel_size = bbox_max_size / 32.

        data.append((model_dir.name, extents, bbox_max_size, voxel_size))

    df = pandas.DataFrame(
        data=data, columns=['name', 'extents', 'bbox_max_size', 'voxel_size']
    )
    csv_file = here / 'data/voxel_size.csv'
    df.to_csv(str(csv_file))
    print(f'Saved to: {csv_file}')

    argsort = df['voxel_size'].argsort()[::-1]
    df = df.iloc[argsort]
    top_images = [top_images[i] for i in argsort]

    print(df)

    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(2, 1)

    df.plot.bar(
        x='name',
        y='voxel_size',
        color=(0.1, 0.1, 0.1, 0.1),
        edgecolor='red',
        rot=45,
        ax=axes[0],
    )
    axes[0].set_xlabel(None)

    top_images = imgviz.tile(top_images, shape=(1, len(top_images)))
    axes[1].imshow(top_images)
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)

    plt.suptitle('Voxel size of YCB_Video_Models')
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
