import numpy as np

import objslampp


class DatasetBase(objslampp.datasets.DatasetBase):

    def _get_invalid_data(self):
        return dict(
            class_id=-1,
            rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            quaternion_true=np.zeros((4,), dtype=np.float64),
            translation_true=np.zeros((3,), dtype=np.float64),
            translation_rough=np.zeros((3,), dtype=np.float64),
        )
