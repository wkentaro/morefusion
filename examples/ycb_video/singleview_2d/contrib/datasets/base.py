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

    def get_example(self, index):
        examples = self.get_examples(index)

        class_ids = [e['class_id'] for e in examples]

        if self._class_ids is None:
            class_id = np.random.choice(class_ids)
        else:
            options = set(self._class_ids) & set(class_ids)
            if options:
                class_id = np.random.choice(list(options))
            else:
                return self._get_invalid_data()
        instance_index = np.random.choice(np.where(class_ids == class_id)[0])

        return examples[instance_index]
