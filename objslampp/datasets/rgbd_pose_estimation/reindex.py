import json

import concurrent.futures
import numpy as np
import tqdm


def task(reindexed_root_dir, dataset, index):
    image_id = dataset._ids[index]
    examples = dataset.get_example(index)
    id_to_class_id = {}
    for i_example, example in enumerate(examples):
        instance_id = f'{image_id}/{i_example:08d}'
        npz_file = reindexed_root_dir / f'{instance_id}.npz'
        npz_file.parent.makedirs_p()
        np.savez_compressed(npz_file, **example)
        id_to_class_id[instance_id] = example['class_id']
    return id_to_class_id


def reindex(reindexed_root_dir: str, datasets: list):
    print(f'Re-indexing following datasets to: {reindexed_root_dir}:')
    for dataset in datasets:
        print(f'  - {dataset}')

    id_to_class_id: dict = {}

    executor = concurrent.futures.ProcessPoolExecutor()

    for dataset in datasets:
        futures = []
        for index in range(len(dataset)):
            future = executor.submit(task, reindexed_root_dir, dataset, index)
            futures.append(future)

        for index, future in tqdm.tqdm(enumerate(futures), total=len(futures)):
            for id, class_id in future.result().items():
                id_to_class_id[id] = int(class_id)

    with open(reindexed_root_dir / 'id_to_class_id.json', 'w') as f:
        json.dump(id_to_class_id, f, indent=4)
