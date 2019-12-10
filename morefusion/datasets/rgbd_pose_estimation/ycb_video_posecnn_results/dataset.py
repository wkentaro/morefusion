import numpy as np

from ...ycb_video import YCBVideoModels
from ...ycb_video import YCBVideoPoseCNNResultsDataset
from ..base import RGBDPoseEstimationDatasetBase


class YCBVideoPoseCNNResultsRGBDPoseEstimationDataset(
    RGBDPoseEstimationDatasetBase
):

    _root_dir = YCBVideoPoseCNNResultsDataset._root_dir

    def __init__(
        self,
        class_ids=None,
    ):
        super().__init__(
            models=YCBVideoModels(),
            class_ids=class_ids,
        )
        self._dataset = YCBVideoPoseCNNResultsDataset()
        self._ids = self._dataset._ids

    def get_frame(self, index):
        frame = self._dataset.get_example(index)

        class_ids = frame['meta']['cls_indexes'].astype(np.int32)
        instance_ids = class_ids.copy()
        T_cam2world = frame['meta']['rotation_translation_matrix']
        T_cam2world = np.r_[T_cam2world, [[0, 0, 0, 1]]].astype(float)
        n_instance = len(instance_ids)
        Ts_cad2cam = np.zeros((n_instance, 4, 4), dtype=float)
        for i in range(n_instance):
            T_cad2cam = frame['meta']['poses'][:, :, i]
            T_cad2cam = np.r_[T_cad2cam, [[0, 0, 0, 1]]]
            Ts_cad2cam[i] = T_cad2cam
        return dict(
            instance_ids=instance_ids,
            class_ids=class_ids,
            rgb=frame['color'],
            depth=frame['depth'],
            instance_label=frame['result']['labels'],
            intrinsic_matrix=frame['meta']['intrinsic_matrix'],
            T_cam2world=T_cam2world,
            Ts_cad2cam=Ts_cad2cam,
            cad_files={},
        )
