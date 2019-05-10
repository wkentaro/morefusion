#!/usr/bin/env python

import numpy as np

import objslampp


class Dataset(objslampp.datasets.DatasetBase):

    def __init__(self, root_dir):
        self._root_dir = root_dir

        self._ids = []
        for video_dir in sorted(self.root_dir.dirs()):
            for npz_file in sorted(video_dir.files()):
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

            instance_index = np.where(example['class_ids'] == 2)[0][0]
            T_cad2cam = example['Ts_cad2cam'][instance_index]
            class_id = example['class_ids'][instance_index]

            cad_file = objslampp.datasets.YCBVideoModels().get_cad_model(
                class_id
            )
            rgb_rend, _, mask_rend = objslampp.extra.pybullet.render_cad(
                cad_file, T_cad2cam, fovy=45, height=480, width=640
            )
            mask_rend = imgviz.label2rgb(mask_rend.astype(int), rgb)

            img = imgviz.tile([
                rgb,
                self._depth2rgb(example['depth']),
                imgviz.label2rgb(example['instance_label'] + 1, rgb),
                imgviz.label2rgb(example['class_label'], rgb),
                rgb_rend,
                mask_rend,
            ], (2, 3), border=(255, 255, 255))
            img = imgviz.resize(img, width=1500)
            return img

    imgviz.io.pyglet_imshow(Images(dataset))
    imgviz.io.pyglet_run()
