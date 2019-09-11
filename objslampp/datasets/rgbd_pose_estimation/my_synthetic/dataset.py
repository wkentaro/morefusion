import numpy as np

from ...ycb_video import YCBVideoModels
from ..base import RGBDPoseEstimationDatasetBase


class MySyntheticRGBDPoseEstimationDataset(RGBDPoseEstimationDatasetBase):

    def __init__(self, root_dir=None, class_ids=None):
        super().__init__(
            models=YCBVideoModels(),
            root_dir=root_dir,
            class_ids=class_ids,
        )
        self._ids = self._get_ids()

    def _get_ids(self):
        ids = []
        for video_dir in sorted(self.root_dir.dirs()):
            for npz_file in sorted(video_dir.files()):
                frame_id = f'{npz_file.parent.name}/{npz_file.stem}'
                ids.append(frame_id)
        return ids

    def get_frame(self, index):
        frame_id = self.ids[index]
        npz_file = self.root_dir / f'{frame_id}.npz'
        frame = np.load(npz_file)

        instance_ids = frame['instance_ids']
        class_ids = frame['class_ids']
        Ts_cad2cam = frame['Ts_cad2cam']

        cad_files = {}
        for ins_id in instance_ids:
            cad_file = npz_file.parent / f'models/{ins_id:08d}.obj'
            if cad_file.exists():
                cad_files[ins_id] = cad_file

        n_instance = len(instance_ids)
        assert len(class_ids) == n_instance
        assert len(Ts_cad2cam) == n_instance

        return dict(
            instance_ids=instance_ids,
            class_ids=class_ids,
            rgb=frame['rgb'],
            depth=frame['depth'],
            instance_label=frame['instance_label'],
            intrinsic_matrix=frame['intrinsic_matrix'],
            T_cam2world=frame['T_cam2world'],
            Ts_cad2cam=Ts_cad2cam,
            cad_files=cad_files,
        )
