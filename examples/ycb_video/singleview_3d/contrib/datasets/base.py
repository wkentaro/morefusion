import numpy as np

import objslampp


class DatasetBase(objslampp.datasets.DatasetBase):

    voxel_dim = 32
    _cache_pitch = {}

    def _get_invalid_data(self):
        return dict(
            class_id=-1,
            pitch=0.,
            rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            pcd=np.zeros((256, 256, 3), dtype=np.float64),
            quaternion_true=np.zeros((4,), dtype=np.float64),
            translation_true=np.zeros((3,), dtype=np.float64),
        )

    def _get_pitch(self, class_id):
        if class_id in self._cache_pitch:
            return self._cache_pitch[class_id]

        models = objslampp.datasets.YCBVideoModels()
        cad_file = models.get_model(class_id=class_id)['textured_simple']
        bbox_diagonal = models.get_bbox_diagonal(mesh_file=cad_file)
        pitch = 1. * bbox_diagonal / self.voxel_dim
        pitch = pitch.astype(np.float32)

        self._cache_pitch[class_id] = pitch
        return pitch
