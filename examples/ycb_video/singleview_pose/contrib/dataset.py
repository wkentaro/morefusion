import pathlib

import imgviz
import numpy as np
import trimesh.transformations as tf

import objslampp


class Dataset(objslampp.datasets.YCBVideoDataset):

    def __init__(self, split, class_ids=None):
        self._class_ids = class_ids
        super(Dataset, self).__init__(split=split, sampling=15)

    def get_ids(
        self,
        split: str,
        sampling: int = 1,
    ):
        assert split in ('train', 'val')

        video2class_ids: dict = {}
        imageset_file: pathlib.Path = self.root_dir / f'image_sets/{split}.txt'
        with open(imageset_file) as f:
            ids: list = []
            for line in f:
                image_id = line.strip()
                video_id, frame_id = image_id.split('/')
                if int(frame_id) % sampling == 0:
                    if video_id in video2class_ids:
                        class_ids = video2class_ids[video_id]
                    else:
                        frame = self.get_frame(image_id)
                        class_ids = frame['meta']['cls_indexes']
                        video2class_ids[video_id] = class_ids
                    ids += [
                        (image_id, class_id) for class_id in class_ids
                        if self._class_ids is None or
                        class_id in self._class_ids
                    ]
            return tuple(ids)

    def get_example(self, index):
        image_id, class_id = self._ids[index]
        frame = self.get_frame(image_id)

        class_ids = frame['meta']['cls_indexes']
        assert class_id in class_ids
        instance_id = np.where(class_ids == class_id)[0][0]

        mask = frame['label'] == class_id
        bbox = objslampp.geometry.masks_to_bboxes([mask])[0]
        y1, x1, y2, x2 = bbox.round().astype(int)

        rgb = frame['color'].copy()
        rgb[~mask] = 0
        rgb = rgb[y1:y2, x1:x2]
        rgb = imgviz.centerize(rgb, (256, 256))

        depth = frame['depth']
        K = frame['meta']['intrinsic_matrix']
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )
        translation_rough = np.nanmean(pcd[mask], axis=0)

        T_cad2cam = frame['meta']['poses'][:, :, instance_id]
        quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
        translation_true = tf.translation_from_matrix(T_cad2cam)

        model_dataset = objslampp.datasets.YCBVideoModelsDataset()
        cad_pcd_file = model_dataset.get_model(class_id=class_id)['points_xyz']
        cad_pcd = np.loadtxt(cad_pcd_file)

        return (
            cad_pcd,
            rgb,
            quaternion_true,
            translation_true,
            translation_rough,
        )
