import json

import concurrent.futures
import numpy as np
import tqdm

from .dataset import YCBVideoRGBDPoseEstimationDataset


dataset_train = YCBVideoRGBDPoseEstimationDataset(split='train', sampling=1)
dataset_val = YCBVideoRGBDPoseEstimationDataset(split='val', sampling=1)
root_dir = dataset_train.root_dir + '.reindexed'


def task(index):
    is_real, image_id = dataset._ids[index]
    examples = dataset.get_example(index)
    id_to_class_id = {}
    for ind, example in enumerate(examples):
        if is_real:
            id = f'data/{image_id}/{ind:08d}'
        else:
            id = f'data_syn/{image_id}/{ind:08d}'
        npz_file = root_dir / f'{id}.npz'
        npz_file.parent.makedirs_p()
        np.savez_compressed(npz_file, **example)
        id_to_class_id[id] = example['class_id']
    return id_to_class_id


def main():
    global dataset

    executor = concurrent.futures.ProcessPoolExecutor()
    futures = []
    for dataset in [dataset_train, dataset_val]:
        for index in range(len(dataset)):
            future = executor.submit(task, index)
            futures.append(future)

    id_to_class_id = {}
    for index, future in tqdm.tqdm(enumerate(futures), total=len(futures)):
        for id, class_id in future.result().items():
            id_to_class_id[id] = int(class_id)

    with open(root_dir / 'id_to_class_id.json', 'w') as f:
        json.dump(id_to_class_id, f, indent=4)


if __name__ == '__main__':
    main()
