#!/usr/bin/env python

import imgviz

import contrib


def main():
    dataset = contrib.datasets.YCBVideoDataset(
        'train',
        class_ids=None,
    )
    print(f'dataset_size: {len(dataset)}')

    # -------------------------------------------------------------------------

    def images():
        for i in range(0, len(dataset)):
            examples = dataset[i]
            for example in examples:
                print(f"class_id: {example['class_id']}")
                print(f"quaternion_true: {example['quaternion_true']}")
                print(f"translation_true: {example['translation_true']}")
                if example['class_id'] > 0:
                    viz = imgviz.tile([
                        example['rgb'],
                        imgviz.depth2rgb(example['pcd'][:, :, 0]),
                        imgviz.depth2rgb(example['pcd'][:, :, 1]),
                        imgviz.depth2rgb(example['pcd'][:, :, 2]),
                    ], (1, 4), border=(255, 255, 255))
                    yield viz

    imgviz.io.pyglet_imshow(images())
    imgviz.io.pyglet_run()


if __name__ == '__main__':
    main()
