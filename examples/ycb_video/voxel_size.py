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


class MainApp(object):

    def _get_data(self):
        class_names = objslampp.datasets.ycb_video.class_names
        models = objslampp.datasets.YCBVideoModelsDataset()

        top_images = []
        data = []
        for model_dir in tqdm.tqdm(sorted(models.root_dir.iterdir())):
            cad_file = model_dir / f'textured_simple.obj'

            top_image = objslampp.sim.pybullet.get_top_image(cad_file)
            top_images.append(top_image)

            cad = trimesh.load(str(cad_file), file_type='obj', process=False)

            name = model_dir.name
            ycb_class_id = int(name.split('_')[0])
            ycb_video_class_id = class_names.index(name)
            extents = cad.bounding_box.extents
            bbox_max_size = np.sqrt((extents ** 2).sum())
            voxel_size = bbox_max_size / 32.

            data.append((
                ycb_class_id,
                ycb_video_class_id,
                name,
                extents,
                bbox_max_size,
                voxel_size,
            ))

        df = pandas.DataFrame(
            data=data,
            columns=[
                'ycb_class_id',
                'ycb_video_class_id',
                'name',
                'extents',
                'bbox_max_size',
                'voxel_size',
            ],
        )
        csv_file = here / 'data/voxel_size.csv'
        df.to_csv(str(csv_file))
        print(f'Saved to: {csv_file}')

        argsort = df['voxel_size'].argsort()[::-1]
        df = df.iloc[argsort]
        top_images = [top_images[i] for i in argsort]

        print(df)
        return df, top_images

    def _plot(self, y='voxel_size', color='red'):
        df, top_images = self._get_data()

        fig, axes = plt.subplots(2, 1)

        df.plot.bar(
            x='name',
            y=y,
            color=(0.1, 0.1, 0.1, 0.1),
            edgecolor=color,
            rot=45,
            ax=axes[0],
        )
        axes[0].set_xlabel(None)

        top_images = imgviz.tile(top_images, shape=(1, len(top_images)))
        axes[1].imshow(top_images)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)

        plt.suptitle(f'{y.capitalize()} of YCB_Video_Models')
        # plt.tight_layout()
        plt.show()

    def plot_voxel_size(self):
        self._plot(y='voxel_size', color='red')

    def plot_bbox_max_size(self):
        self._plot(y='bbox_max_size', color='blue')


if __name__ == '__main__':
    import fire

    fire.Fire(MainApp)
