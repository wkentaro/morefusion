#!/usr/bin/env python

import pathlib

import numpy as np

import objslampp


class Dataset(objslampp.datasets.DatasetBase):

    def __init__(self, root_dir):
        self._root_dir = pathlib.Path(root_dir)

        self._ids = []
        for video_dir in sorted(self.root_dir.iterdir()):
            for npz_file in sorted(video_dir.iterdir()):
                frame_id = f'{npz_file.parent.name}/{npz_file.stem}'
                self._ids.append(frame_id)

    def get_example(self, index):
        frame_id = self.ids[index]
        npz_file = self.root_dir / f'{frame_id}.npz'
        data = np.load(npz_file)
        return dict(data)


if __name__ == '__main__':
    import argparse
    import imgviz

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('root_dir', help='root dir')
    args = parser.parse_args()

    dataset = Dataset(root_dir=args.root_dir)

    class Images:

        def __init__(self, dataset):
            self._dataset = dataset
            self._depth2rgb = imgviz.Depth2RGB()

        def __len__(self):
            return len(self._dataset)

        def __getitem__(self, i):
            print(f'[{i:08d}] [{dataset.ids[i]}]')
            example = dataset[i]
            rgb = example['rgb']
            img = imgviz.tile([
                rgb,
                self._depth2rgb(example['depth']),
                imgviz.label2rgb(example['instance_label'] + 1, rgb),
                imgviz.label2rgb(example['class_label'], rgb),
            ], border=(255, 255, 255))
            return img

    imgviz.io.pyglet_imshow(Images(dataset))
    imgviz.io.pyglet_run()
