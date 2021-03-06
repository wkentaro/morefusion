#!/usr/bin/env python

import concurrent.futures

import imgviz
import matplotlib.pyplot as plt
import pandas
import pybullet  # NOQA

import morefusion


def get_data():
    models = morefusion.datasets.YCBVideoModels()
    class_ids = list(range(models.n_class))[1:]

    cad_files = [models.get_cad_file(class_id=c) for c in class_ids]

    with concurrent.futures.ProcessPoolExecutor() as p:
        top_images = []
        for cad_file in cad_files:
            top_images.append(
                p.submit(morefusion.extra.pybullet.get_top_image, cad_file)
            )

        bbox_diagonals = []
        for class_id in class_ids:
            bbox_diagonals.append(
                p.submit(models.get_bbox_diagonal, class_id=class_id)
            )
    top_images = [future.result() for future in top_images]
    bbox_diagonals = [future.result() for future in bbox_diagonals]

    data = []
    for class_id, bbox_diagonal in zip(class_ids, bbox_diagonals):
        class_name = models.class_names[class_id]
        ycb_class_id = int(class_name.split("_")[0])
        ycb_video_class_id = class_id
        voxel_size = bbox_diagonal / 32.0
        data.append(
            (
                ycb_class_id,
                ycb_video_class_id,
                class_name,
                bbox_diagonal,
                voxel_size,
            )
        )

    df = pandas.DataFrame(
        data=data,
        columns=[
            "ycb_class_id",
            "ycb_video_class_id",
            "name",
            "bbox_diagonal",
            "voxel_size",
        ],
    )
    print(df)

    argsort = df["voxel_size"].argsort()[::-1]
    df = df.iloc[argsort]
    top_images = [top_images[i] for i in argsort]

    return df, top_images


def main():
    df, top_images = get_data()

    fig = plt.figure(figsize=(15, 11))
    axes = fig.subplots(3, 1)

    df.plot.bar(
        x="name",
        y="bbox_diagonal",
        color=(0.1, 0.1, 0.1, 0.1),
        edgecolor="red",
        ax=axes[0],
    )
    axes[0].get_xaxis().set_visible(False)

    df.plot.bar(
        x="name",
        y="voxel_size",
        color=(0.1, 0.1, 0.1, 0.1),
        edgecolor="blue",
        rot=45,
        ax=axes[1],
    )
    axes[1].set_xlabel(None)

    axes[2].imshow(imgviz.tile(top_images, shape=(1, len(top_images))))
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)

    plt.suptitle("BBox and Voxel size of YCB_Video_Models")

    plt.show()


if __name__ == "__main__":
    main()
