import pathlib

import imgviz
import numpy as np
import trimesh
import trimesh.transformations as tf

from .. import geometry
from .camera2cad import Camera2CAD
from .ycb_video import YCBVideoDataset
from .ycb_video_models import YCBVideoModelsDataset


class YCBVideoMultiViewAlignmentDataset(YCBVideoDataset):

    roi_size = 256
    voxel_dim = 32

    def __init__(
        self,
        split,
        sampling=15,
        class_ids=None,
        num_frames_scan=None,
    ):
        self._class_ids = class_ids
        super(YCBVideoMultiViewAlignmentDataset, self).__init__(
            split=split, sampling=sampling
        )
        self._cache_cad_data = {}
        self._cache_pitch = {}

        if num_frames_scan is None:
            num_frames_scan = 10
        self._num_frames_scan = num_frames_scan

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
            scan_origin, T_cad2cam, scan_rgbs, scan_pcds, scan_masks = \
                self._get_scan_data(image_id, class_id, random_state)
        except ValueError:
            valid = 0  # indicates skipped while training
            scan_origin = np.zeros((), dtype=np.float32)
            scan_rgbs = np.zeros((), dtype=np.float32)
            scan_pcds = np.zeros((), dtype=np.float32)
            scan_masks = np.zeros((), dtype=np.float32)
            T_cad2cam = np.zeros((), dtype=np.float32)

        if valid:
            cad_origin, cad_rgbs, cad_pcds, cad_points = self._get_cad_data(
                class_id
            )
        else:
            cad_origin = np.zeros((), dtype=np.float32)
            cad_rgbs = np.zeros((), dtype=np.float32)
            cad_pcds = np.zeros((), dtype=np.float32)
            cad_points = np.zeros((), dtype=np.float32)

        if valid:
            T_cam2cad = tf.inverse_matrix(T_cad2cam)
            cam2cad = Camera2CAD(
                self.voxel_dim, pitch, cad_origin, scan_origin
            )
            quaternion_target, translation_target = cam2cad.target_from_matrix(
                T_cam2cad
            )
        else:
            quaternion_target = np.zeros((), dtype=np.float32)
            translation_target = np.zeros((), dtype=np.float32)

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

            cad_points=cad_points,
            quaternion_target=quaternion_target,
            translation_target=translation_target,
        )

    def _get_cad_data(self, class_id):
        if class_id in self._cache_cad_data:
            return self._cache_cad_data[class_id]

        pitch = self._get_pitch(class_id)
        origin = np.array(
            (- self.voxel_dim // 2 * pitch,) * 3, dtype=np.float32
        )

        models = YCBVideoModelsDataset()
        cad_file = models.get_model(class_id=class_id)['textured_simple']
        K, Ts_cam2world, rgbs, depths, segms = models.get_spherical_views(
            cad_file
        )

        # transform point cloud to world frame
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

        rgbs, pcds, _ = self._to_roi_image(rgbs, pcds, segms == 0)

        points_file = models.get_model(class_id=class_id)['points_xyz']
        points = np.loadtxt(points_file).astype(np.float32)

        assert isinstance(origin, np.ndarray)
        assert origin.dtype == np.float32
        assert isinstance(rgbs, np.ndarray)
        assert rgbs.dtype == np.uint8
        assert isinstance(pcds, np.ndarray)
        assert pcds.dtype == np.float32
        assert isinstance(points, np.ndarray)
        assert points.dtype == np.float32
        assert points.shape == (points.shape[0], 3)

        self._cache_cad_data[class_id] = (origin, rgbs, pcds, points)
        return origin, rgbs, pcds, points

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

    def _to_roi_image(self, rgbs, pcds, masks):
        assert rgbs.shape[:3] == pcds.shape[:3] == masks.shape[:3]

        bboxes = geometry.masks_to_bboxes(masks).round().astype(int)

        N = len(bboxes)
        roi_size = self.roi_size
        rgbs_roi = np.zeros((N, roi_size, roi_size, 3), dtype=rgbs.dtype)
        pcds_roi = np.zeros((N, roi_size, roi_size, 3), dtype=pcds.dtype)
        masks_roi = np.zeros((N, roi_size, roi_size), dtype=masks.dtype)

        for i in range(len(bboxes)):
            y1, x1, y2, x2 = bboxes[i]
            rgb = rgbs[i]
            pcd = pcds[i]
            mask = masks[i]

            rgb = rgb[y1:y2, x1:x2]
            pcd = pcd[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]

            rgb = imgviz.centerize(rgb, shape=(roi_size, roi_size), cval=255)
            pcd = imgviz.centerize(
                pcd, shape=(roi_size, roi_size), cval=np.nan
            )
            mask = mask.astype(np.float32)
            mask = imgviz.centerize(mask, shape=(roi_size, roi_size), cval=0)
            mask = mask > 0.5

            rgbs_roi[i] = rgb
            pcds_roi[i] = pcd
            masks_roi[i] = mask

        return rgbs_roi, pcds_roi, masks_roi

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
        mapping = geometry.VoxelMapping(pitch=pitch, voxel_size=self.voxel_dim)
        origin = aabb_center - mapping.voxel_bbox_extents / 2
        origin = origin.astype(np.float32)

        # transformation: cad frame to camera frame
        T_cad2cam = frame['meta']['poses'][:, :, instance_id]
        T_cad2cam = np.r_[T_cad2cam, [[0, 0, 0, 1]]]
        T_cad2cam = T_cad2cam.astype(np.float32)

        # ---------------------------------------------------------------------

        scene_id, frame_id = image_id.split('/')
        frame_ids = [f'{i:06d}' for i in range(1, int(frame_id))]
        n_frames_prev = self._num_frames_scan - 1
        if len(frame_ids) > n_frames_prev:
            indices = random_state.permutation(len(frame_ids))[:n_frames_prev]
            frame_ids = [frame_ids[i] for i in sorted(indices)]
        frame_ids += [frame_id]

        # transform point cloud to camera0 frame
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

        rgbs, pcds, masks = self._to_roi_image(rgbs, pcds, masks)

        assert isinstance(origin, np.ndarray)
        assert origin.dtype == np.float32
        assert isinstance(T_cad2cam, np.ndarray)
        assert T_cad2cam.dtype == np.float32
        assert isinstance(rgbs, np.ndarray)
        assert rgbs.dtype == np.uint8
        assert isinstance(pcds, np.ndarray)
        assert pcds.dtype == np.float32
        assert isinstance(masks, np.ndarray)
        assert masks.dtype == bool

        return origin, T_cad2cam, rgbs, pcds, masks
