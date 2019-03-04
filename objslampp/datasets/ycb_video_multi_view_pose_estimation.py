import numpy as np
import trimesh

from .. import geometry
from .ycb_video import YCBVideoDataset
from .ycb_video_models import YCBVideoModelsDataset


class YCBVideoMultiViewPoseEstimationDataset(YCBVideoDataset):

    voxel_dim = 32

    def __init__(self, *args, **kwargs):
        super(YCBVideoMultiViewPoseEstimationDataset, self).__init__(
            *args, **kwargs
        )
        self._cache_cad_data = {}
        self._cache_pitch = {}

    def get_cad_data(self, class_id):
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
        pcds = np.asarray(pcds)

        pitch = self._get_pitch(class_id)
        origin = np.array(
            (- self.voxel_dim // 2 * pitch,) * 3, dtype=float
        )

        self._cache_cad_data[class_id] = (origin, rgbs, pcds)
        return origin, rgbs, pcds

    def _get_pitch(self, class_id):
        if class_id in self._cache_pitch:
            return self._cache_pitch[class_id]

        models = YCBVideoModelsDataset()
        cad_file = models.get_model(class_id=class_id)['textured_simple']
        bbox_diagonal = models.get_bbox_diagonal(mesh_file=cad_file)
        pitch = 1. * bbox_diagonal / self.voxel_dim

        self._cache_pitch[class_id] = pitch
        return pitch

    def get_scan_data(self, image_id):
        frame = self.get_frame(image_id)
        T_world2cam = np.r_[
            frame['meta']['rotation_translation_matrix'],
            [[0, 0, 0, 1]],
        ]
        T_cam2world = np.linalg.inv(T_world2cam)
        K = frame['meta']['intrinsic_matrix']
        depth = frame['depth']
        pcd = geometry.pointcloud_from_depth(
            depth, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2]
        )
        isnan = np.isnan(pcd).any(axis=2)
        pcd[~isnan] = trimesh.transform_points(pcd[~isnan], T_cam2world)

        class_ids = frame['meta']['cls_indexes']
        origins = []
        gt_poses = []
        for instance_id, class_id in enumerate(class_ids):
            pitch = self._get_pitch(class_id=class_id)

            mask = frame['label'] == class_id
            pcd_ins = pcd[mask & (~isnan)]
            aabb_min, aabb_max = geometry.get_aabb_from_points(pcd_ins)
            aabb_extents = aabb_max - aabb_min
            aabb_center = aabb_extents / 2 + aabb_min
            mapping = geometry.VoxelMapping(pitch=pitch, voxel_size=32)
            origin = aabb_center - mapping.voxel_bbox_extents / 2
            origins.append(origin)

            gt_pose = frame['meta']['poses'][:, :, instance_id]
            gt_pose = np.r_[gt_pose, [[0, 0, 0, 1]]]
            gt_pose = T_cam2world @ gt_pose
            gt_poses.append(gt_pose)

        # ---------------------------------------------------------------------

        scene_id, frame_id = image_id.split('/')
        frame_ids = [f'{i:06d}' for i in range(1, int(frame_id) + 1)]
        frame_ids = np.random.choice(frame_ids, 10)

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
            isnan = np.isnan(pcd).any(axis=2)
            pcd[~isnan] = trimesh.transform_points(pcd[~isnan], T_cam2world)
            pcds.append(pcd)

        rgbs = np.asarray(rgbs)
        pcds = np.asarray(pcds)
        labels = np.asarray(labels)

        return class_ids, origins, gt_poses, rgbs, pcds, labels

    def __getitem__(self, index: int):
        image_id = self.imageset[index]

        try:
            class_ids, scan_origins, gt_poses, scan_rgbs, \
                scan_pcds, scan_labels = self.get_scan_data(image_id)
        except Exception:
            return dict(
                class_id=-1,
                pitch=np.empty(()),

                cad_origin=np.empty(()),
                cad_rgbs=np.empty(()),
                cad_pcds=np.empty(()),

                scan_rgbs=np.empty(()),
                scan_pcds=np.empty(()),
                scan_masks=np.empty(()),
                scan_origin=np.empty(()),
                gt_pose=np.empty(()),
            )

        instance_id = np.random.randint(0, len(class_ids))
        class_id = class_ids[instance_id]
        pitch = self._get_pitch(class_id=class_id)

        cad_origin, cad_rgbs, cad_pcds = self.get_cad_data(class_id)

        scan_masks = scan_labels == class_id
        scan_origin = scan_origins[instance_id]
        gt_pose = gt_poses[instance_id]

        return dict(
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
        )
