#!/usr/bin/env python

import concurrent.futures
import pathlib

import imgviz
import matplotlib.pyplot as plt
import pandas
import pybullet  # NOQA

import objslampp


here = pathlib.Path(__file__).resolve().parent


class MainApp(object):

    def _get_data(self):
        class_names = objslampp.datasets.ycb_video.class_names
        models = objslampp.datasets.YCBVideoModelsDataset()

        cad_files = [
            models.get_model(class_name=name)['textured_simple']
            for name in class_names[1:]
        ]

        with concurrent.futures.ProcessPoolExecutor() as p:
            top_images = []
            for cad_file in cad_files:
                top_images.append(
                    p.submit(objslampp.sim.pybullet.get_top_image, cad_file)
                )

            bbox_diagonals = []
            for cad_file in cad_files:
                bbox_diagonals.append(
                    p.submit(models.get_bbox_diagonal, mesh_file=cad_file)
                )
        top_images = [future.result() for future in top_images]
        bbox_diagonals = [future.result() for future in bbox_diagonals]

        data = []
        for class_name, bbox_diagonal in zip(class_names[1:], bbox_diagonals):
            ycb_class_id = int(class_name.split('_')[0])
            ycb_video_class_id = class_names.index(class_name)
            voxel_size = bbox_diagonal / 32.
            data.append((
                ycb_class_id,
                ycb_video_class_id,
                class_name,
                bbox_diagonal,
                voxel_size,
            ))

        df = pandas.DataFrame(
            data=data,
            columns=[
                'ycb_class_id',
                'ycb_video_class_id',
                'name',
                'bbox_diagonal',
                'voxel_size',
            ],
        )
        print(df)

        argsort = df['voxel_size'].argsort()[::-1]
        df = df.iloc[argsort]
        top_images = [top_images[i] for i in argsort]

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

    def plot_bbox_diagonal(self):
        self._plot(y='bbox_diagonal', color='blue')


if __name__ == '__main__':
    import fire

    fire.Fire(MainApp)
