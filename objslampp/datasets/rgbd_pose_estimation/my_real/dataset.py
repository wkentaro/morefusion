import imgviz
import numpy as np
import yaml
import trimesh.transformations as tf

from .... import geometry as geometry_module
from ...ycb_video import YCBVideoModels
from ..base import RGBDPoseEstimationDatasetBase


class MyRealRGBDPoseEstimationDataset(RGBDPoseEstimationDatasetBase):

    def __init__(self, root_dir=None, class_ids=None):
        super().__init__(
            models=YCBVideoModels(),
            root_dir=root_dir,
            class_ids=class_ids,
        )
        self._ids = self._get_ids()

    def _get_ids(self):
        ids = self.root_dir.dirs()
        return tuple(ids)

    def get_frame(self, index):
        frame_dir = self.ids[index]

        rgb_file = frame_dir / 'image.png'
        rgb = imgviz.io.imread(rgb_file)[:, :, :3]

        depth_file = frame_dir / 'depth.npz'
        depth = np.load(depth_file)['arr_0']
        depth = depth.astype(np.float32) / 1000.
        depth[depth == 0] = np.nan
        assert rgb.shape[:2] == depth.shape

        detections_file = frame_dir / 'detections.npz'
        detections = np.load(detections_file)
        masks = detections['masks']
        class_ids = detections['class_ids']
        # scores = detections['scores']

        camera_info_file = frame_dir / 'camera_info.yaml'
        with open(camera_info_file) as f:
            camera_info = yaml.safe_load(f)
        K = np.array(camera_info['K']).reshape(3, 3)

        instance_ids = []
        instance_label = np.zeros(rgb.shape[:2], dtype=np.int32)
        for i, mask in enumerate(masks):
            instance_id = i + 1
            instance_ids.append(instance_id)
            instance_label[mask] = instance_id
        instance_ids = np.array(instance_ids, dtype=np.int32)

        pcd = geometry_module.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        Ts_cad2cam = []
        for instance_id in instance_ids:
            mask = instance_label == instance_id
            translation_rough = np.nanmean(pcd[mask], axis=0)
            T_cad2cam = tf.translation_matrix(translation_rough)
            Ts_cad2cam.append(T_cad2cam)
        Ts_cad2cam = np.array(Ts_cad2cam, dtype=float)

        return dict(
            instance_ids=instance_ids,
            class_ids=class_ids,
            rgb=rgb,
            depth=depth,
            instance_label=instance_label,
            intrinsic_matrix=K,
            T_cam2world=np.eye(4),
            Ts_cad2cam=Ts_cad2cam,
            cad_files={},
        )
