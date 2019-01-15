import pathlib

import imgviz
import scipy.io


class YCBVideoDataset(object):

    def __init__(self):
        home = pathlib.Path.home()
        self.root_dir = home / 'data/datasets/YCB/YCB_Video_Dataset'

    def imageset(self, split):
        assert split in ['train', 'val', 'trainval']
        imageset_file = self.root_dir / f'image_sets/{split}.txt'
        with open(imageset_file) as f:
            imageset = [l.strip() for l in f.readlines()]
        return imageset

    def get_frame(self, image_id):
        meta_file = self.root_dir / 'data' / (image_id + '-meta.mat')
        meta = scipy.io.loadmat(
            meta_file, squeeze_me=True, struct_as_record=True
        )

        color_file = self.root_dir / 'data' / (image_id + '-color.png')
        color = imgviz.io.imread(color_file)

        depth_file = self.root_dir / 'data' / (image_id + '-depth.png')
        depth = imgviz.io.imread(depth_file)
        depth = depth.astype(float) / meta['factor_depth']
        depth[depth == 0] = float('nan')

        label_file = self.root_dir / 'data' / (image_id + '-label.png')
        label = imgviz.io.imread(label_file)

        return dict(
            meta=meta,
            color=color,
            depth=depth,
            label=label,
        )
