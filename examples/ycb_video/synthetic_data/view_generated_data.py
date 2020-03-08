#!/usr/bin/env python

import argparse

import imgviz
import numpy as np
import pybullet  # NOQA
import trimesh

import morefusion


class Dataset(morefusion.datasets.DatasetBase):

    _models = morefusion.datasets.YCBVideoModels()

    def __init__(self, root_dir):
        self._root_dir = root_dir

        self._ids = []
        for video_dir in sorted(self.root_dir.dirs()):
            frame_ids = []
            for npz_file in sorted(video_dir.files()):
                frame_id = f"{npz_file.parent.name}/{npz_file.stem}"
                frame_ids.append(frame_id)
            self._ids.append(frame_ids)

        self._depth2rgb = imgviz.Depth2RGB()

    def get_examples(self, index):
        for frame_id in self._ids[index]:
            print(f"[{index:08d}] {frame_id}")
            yield self.get_example(frame_id)

    def get_example(self, frame_id):
        npz_file = self.root_dir / f"{frame_id}.npz"
        example = dict(np.load(npz_file))
        example["cad_files"] = {}
        for cad_file in (npz_file.parent / "models").glob("*"):
            ins_id = int(cad_file.basename().stem)
            example["cad_files"][ins_id] = cad_file
        return example

    def get_scenes(self, index):
        self._depth2rgb = imgviz.Depth2RGB()
        for example in self.get_examples(index):
            yield self.get_scene(example)

    def get_scene(self, example):
        rgb = example["rgb"]
        gray = imgviz.color.gray2rgb(imgviz.color.rgb2gray(rgb))

        try:
            instance_index = np.where(example["class_ids"] != 0)[0][0]
        except IndexError:
            instance_index = None

        if instance_index is not None:
            T_cad2cam = example["Ts_cad2cam"][instance_index]
            class_id = example["class_ids"][instance_index]

            cad_file = self._models.get_cad_file(class_id)
            _, _, mask_rend = morefusion.extra.pybullet.render_cad(
                cad_file, T_cad2cam, fovy=45, height=480, width=640
            )
            mask_rend = imgviz.label2rgb(
                mask_rend.astype(np.uint8), gray, alpha=0.7
            )
        else:
            mask_rend = np.zeros_like(rgb)

        # scene
        scene = trimesh.Scene()
        K = example["intrinsic_matrix"]
        T_cam2world = example["T_cam2world"]
        pcd = morefusion.geometry.pointcloud_from_depth(
            example["depth"], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )
        nonnan = ~np.isnan(example["depth"])
        geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
        scene.add_geometry(geom, transform=T_cam2world)
        for ins_id, cls_id, T_cad2cam in zip(
            example["instance_ids"],
            example["class_ids"],
            example["Ts_cad2cam"],
        ):
            if ins_id == 0:
                # ground plane
                continue
            if cls_id == 0:
                cad_file = example["cad_files"][ins_id]
                cad = trimesh.load_mesh(cad_file, process=False)
            else:
                cad = self._models.get_cad(class_id=cls_id)
                if hasattr(cad.visual, "to_color"):
                    cad.visual = cad.visual.to_color()
            scene.add_geometry(
                cad,
                node_name=str(ins_id),
                geom_name=str(cls_id),
                transform=T_cam2world @ T_cad2cam,
            )

        scene.camera.resolution = rgb.shape[1], rgb.shape[0]
        scene.camera.focal = K[0, 0], K[1, 1]
        scene.camera_transform = morefusion.extra.trimesh.to_opengl_transform(
            T_cam2world
        )

        ins_viz = imgviz.label2rgb(
            example["instance_label"] + 1, rgb, alpha=0.7
        )
        cls_viz = imgviz.label2rgb(example["class_label"], rgb, alpha=0.7)

        return {
            "rgb": rgb,
            "depth": self._depth2rgb(example["depth"]),
            "ins": ins_viz,
            "cls": cls_viz,
            "mask_rend": mask_rend,
            "scene": scene,
        }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir", help="root dir")
    args = parser.parse_args()

    dataset = Dataset(root_dir=args.root_dir)

    morefusion.extra.trimesh.display_scenes(
        (dataset.get_scenes(index) for index in range(len(dataset))),
        tile=(2, 3),
    )


if __name__ == "__main__":
    main()
