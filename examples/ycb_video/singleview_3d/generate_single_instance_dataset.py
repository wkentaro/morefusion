#!/usr/bin/env python

import json

import chainer
import concurrent.futures
import path
import numpy as np
import tqdm

import contrib


dataset = None
dataset_dir = chainer.dataset.get_dataset_directory(
    'wkentaro/objslampp/ycb_video/single_instance_dataset'
)
dataset_dir = path.Path(dataset_dir)


def task(index):
    is_real, image_id = dataset._ids[index]
    examples = dataset.get_example(index)
    id_to_class_id = {}
    for ind, example in enumerate(examples):
        if is_real:
            id = f'data/{image_id}/{ind:08d}'
        else:
            id = f'data_syn/{image_id}/{ind:08d}'
        npz_file = dataset_dir / f'{id}.npz'
        npz_file.parent.makedirs_p()
        np.savez_compressed(npz_file, **example)
        id_to_class_id[id] = example['class_id']
    return id_to_class_id


def main():
    global dataset

    id_to_class_id = {}
    for split in ['val', 'train']:
        dataset = contrib.datasets.YCBVideoDataset(split=split, sampling=1)
        executor = concurrent.futures.ProcessPoolExecutor()
        futures = []
        for index in range(len(dataset)):
            future = executor.submit(task, index)
            futures.append(future)
        for index, future in tqdm.tqdm(enumerate(futures), total=len(futures)):
            for id, class_id in future.result().items():
                id_to_class_id[id] = int(class_id)
        del dataset
    with open(dataset_dir / 'id_to_class_id.json', 'w') as f:
        json.dump(id_to_class_id, f, indent=4)


if __name__ == '__main__':
    main()
