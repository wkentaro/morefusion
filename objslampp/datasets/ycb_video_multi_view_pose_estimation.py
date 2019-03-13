import pathlib

import numpy as np
import trimesh
import trimesh.transformations as tf

from .. import geometry
from .ycb_video import YCBVideoDataset
from .ycb_video_models import YCBVideoModelsDataset


class YCBVideoMultiViewPoseEstimationDataset(YCBVideoDataset):

    voxel_dim = 32

    def __init__(self, split, sampling=15, class_ids=None):
        self._class_ids = class_ids
        super(YCBVideoMultiViewPoseEstimationDataset, self).__init__(
            split=split, sampling=sampling
        )
        self._cache_cad_data = {}
        self._cache_pitch = {}

        self._random_seeds = None
        if split == 'val':
            random_state = np.random.RandomState(1234)
            self._random_seeds = random_state.randint(0, 9999, len(self.ids))

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

    def get_example(self, i):
        valid = 1

        image_id, class_id = self.ids[i]

        pitch = self._get_pitch(class_id=class_id)

        try:
            random_state = None
            if self._random_seeds is not None:
                random_state = np.random.RandomState(
                    seed=self._random_seeds[i]
                )
            scan_origin, gt_pose, scan_rgbs, scan_pcds, scan_masks = \
                self._get_scan_data(image_id, class_id, random_state)
        except ValueError:
            valid = 0  # indicates skipped while training
            scan_origin = np.zeros((), dtype=np.float32)
            scan_rgbs = np.zeros((), dtype=np.float32)
            scan_pcds = np.zeros((), dtype=np.float32)
            scan_masks = np.zeros((), dtype=np.float32)
            gt_pose = np.zeros((), dtype=np.float32)

        if valid:
            cad_origin, cad_rgbs, cad_pcds = self._get_cad_data(class_id)
        else:
            cad_origin = np.zeros((), dtype=np.float32)
            cad_rgbs = np.zeros((), dtype=np.float32)
            cad_pcds = np.zeros((), dtype=np.float32)

        if valid:
            gt_quaternion = tf.quaternion_from_matrix(gt_pose)
            gt_quaternion = gt_quaternion.astype(np.float32)

            # initial alignment by fitting voxel origins (cad and origin)
            translation_init = scan_origin - cad_origin

            gt_translation = tf.translation_from_matrix(gt_pose)
            gt_translation = gt_translation - translation_init
            gt_translation = gt_translation / pitch / self.voxel_dim
            gt_translation = gt_translation.astype(np.float32)
        else:
            gt_quaternion = np.zeros((), dtype=np.float32)
            gt_translation = np.zeros((), dtype=np.float32)

        video_id, frame_id = image_id.split('/')
        video_id, frame_id = int(video_id), int(frame_id)
        return dict(
            valid=valid,
            video_id=video_id,

            class_id=class_id,
            pitch=pitch,

            cad_origin=cad_origin,
            cad_rgbs=cad_rgbs,
            cad_pcds=cad_pcds,          # cad coordinate

            scan_rgbs=scan_rgbs,
            scan_pcds=scan_pcds,        # world coordinate
            scan_masks=scan_masks,
            scan_origin=scan_origin,    # for current_view, world coordinate

            gt_pose=gt_pose,            # cad -> world
            gt_quaternion=gt_quaternion,
            gt_translation=gt_translation,
        )

    def _get_cad_data(self, class_id):
        if class_id in self._cache_cad_data:
            return self._cache_cad_data[class_id]

        models = YCBVideoModelsDataset()
        cad_file = models.get_model(class_id=class_id)['textured_simple']
        K, Ts_cam2world, rgbs, depths, segms = models.get_spherical_views(
            cad_file
        )

        pcds = []
        for T_cam2world, depth in zip(Ts_cam2world, depths):
            pcd = geometry.pointcloud_from_depth(
                depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
            )  # in camera coord
            isnan = np.isnan(pcd).any(axis=2)
            pcd[~isnan] = trimesh.transform_points(
                pcd[~isnan], T_cam2world
            )  # in world coord
            pcds.append(pcd)
        pcds = np.asarray(pcds, dtype=np.float32)

        pitch = self._get_pitch(class_id)
        origin = np.array(
            (- self.voxel_dim // 2 * pitch,) * 3, dtype=np.float32
        )

        assert isinstance(origin, np.ndarray)
        assert origin.dtype == np.float32
        assert isinstance(rgbs, np.ndarray)
        assert rgbs.dtype == np.uint8
        assert isinstance(pcds, np.ndarray)
        assert pcds.dtype == np.float32

        self._cache_cad_data[class_id] = (origin, rgbs, pcds)
        return origin, rgbs, pcds

    def _get_pitch(self, class_id):
        if class_id in self._cache_pitch:
            return self._cache_pitch[class_id]

        models = YCBVideoModelsDataset()
        cad_file = models.get_model(class_id=class_id)['textured_simple']
        bbox_diagonal = models.get_bbox_diagonal(mesh_file=cad_file)
        pitch = 1. * bbox_diagonal / self.voxel_dim
        pitch = pitch.astype(np.float32)

        assert isinstance(pitch, np.float32)

        self._cache_pitch[class_id] = pitch
        return pitch

    def _get_scan_data(self, image_id, class_id, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()

        frame = self.get_frame(image_id)

        T_world2cam0 = np.r_[
            frame['meta']['rotation_translation_matrix'],
            [[0, 0, 0, 1]],
        ]

        K = frame['meta']['intrinsic_matrix']
        depth = frame['depth']
        pcd = geometry.pointcloud_from_depth(
            depth, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2]
        )

        class_ids = frame['meta']['cls_indexes']
        assert class_id in class_ids

        instance_id = np.where(class_ids == class_id)[0][0]

        pitch = self._get_pitch(class_id=class_id)

        mask = frame['label'] == class_id
        isnan = np.isnan(pcd).any(axis=2)
        pcd_ins = pcd[mask & (~isnan)]
        aabb_min, aabb_max = geometry.get_aabb_from_points(pcd_ins)
        aabb_extents = aabb_max - aabb_min
        aabb_center = aabb_extents / 2 + aabb_min
        mapping = geometry.VoxelMapping(pitch=pitch, voxel_size=32)
        origin = aabb_center - mapping.voxel_bbox_extents / 2
        origin = origin.astype(np.float32)

        # transformation: cad frame to camera frame
        gt_pose = frame['meta']['poses'][:, :, instance_id]
        gt_pose = np.r_[gt_pose, [[0, 0, 0, 1]]]
        # transformation: camera frame to world frame
        gt_pose = gt_pose.astype(np.float32)

        # ---------------------------------------------------------------------

        scene_id, frame_id = image_id.split('/')
        frame_ids = [f'{i:06d}' for i in range(1, int(frame_id))]
        if len(frame_ids) > 9:
            indices = random_state.permutation(len(frame_ids))[:9]
            frame_ids = [frame_ids[i] for i in sorted(indices)]
        frame_ids += [frame_id]

        rgbs = []
        pcds = []
        labels = []
        for frame_id in frame_ids:
            image_id = self.get_image_id(scene_id, frame_id)
            frame = self.get_frame(image_id)

            rgbs.append(frame['color'])
            labels.append(frame['label'])

            K = frame['meta']['intrinsic_matrix']
            depth = frame['depth']
            pcd = geometry.pointcloud_from_depth(
                depth, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2]
            )

            T_world2cam = np.r_[
                frame['meta']['rotation_translation_matrix'],
                [[0, 0, 0, 1]],
            ]
            T_cam2world = np.linalg.inv(T_world2cam)
            T_cam2cam0 = T_world2cam0 @ T_cam2world
            isnan = np.isnan(pcd).any(axis=2)
            pcd[~isnan] = trimesh.transform_points(pcd[~isnan], T_cam2cam0)
            pcds.append(pcd)
        rgbs = np.asarray(rgbs)
        pcds = np.asarray(pcds, dtype=np.float32)
        masks = np.asarray(labels) == class_id

        assert isinstance(origin, np.ndarray)
        assert origin.dtype == np.float32
        assert isinstance(gt_pose, np.ndarray)
        assert gt_pose.dtype == np.float32
        assert isinstance(rgbs, np.ndarray)
        assert rgbs.dtype == np.uint8
        assert isinstance(pcds, np.ndarray)
        assert pcds.dtype == np.float32
        assert isinstance(masks, np.ndarray)
        assert masks.dtype == bool

        return origin, gt_pose, rgbs, pcds, masks
