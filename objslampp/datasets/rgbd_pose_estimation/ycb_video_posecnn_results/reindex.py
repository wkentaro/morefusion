import json

import concurrent.futures
import numpy as np
import tqdm

from .dataset import YCBVideoPoseCNNResultsRGBDPoseEstimationDataset


dataset = YCBVideoPoseCNNResultsRGBDPoseEstimationDataset()

root_dir = dataset.root_dir + '.reindexed.w_full_occupancy'


def task(index):
    image_id = dataset._ids[index]
    examples = dataset.get_example(index)
    id_to_class_id = {}
    for ind, example in enumerate(examples):
        id = f'data/{image_id}/{ind:08d}'
        npz_file = root_dir / f'{id}.npz'
        npz_file.parent.makedirs_p()
        np.savez_compressed(npz_file, **example)
        id_to_class_id[id] = example['class_id']
    return id_to_class_id


def main():
    id_to_class_id = {}
    executor = concurrent.futures.ProcessPoolExecutor()
    futures = []
    for index in range(len(dataset)):
        future = executor.submit(task, index)
        futures.append(future)

    for index, future in tqdm.tqdm(enumerate(futures), total=len(futures)):
        for id, class_id in future.result().items():
            id_to_class_id[id] = int(class_id)

    with open(root_dir / 'id_to_class_id.json', 'w') as f:
        json.dump(id_to_class_id, f, indent=4)


if __name__ == '__main__':
    main()
