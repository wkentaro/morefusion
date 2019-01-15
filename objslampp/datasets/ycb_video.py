import pathlib

import gdown
import imgviz
import scipy.io


class YCBVideoDataset(object):

    root_dir = pathlib.Path.home() / 'data/datasets/YCB/YCB_Video_Dataset'

    def __init__(self):
        pass

    @classmethod
    def download(cls):
        url = 'https://drive.google.com/uc?id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi'  # NOQA
        md5 = None  # 'c9122e177a766a9691cab13c5cda41a9'
        gdown.cached_download(
            url=url,
            path=str(cls.root_dir) + '.zip',
            md5=md5,
            postprocess=gdown.extractall,
        )

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


class YCBVideoModels(object):

    root_dir = pathlib.Path.home() / 'data/datasets/YCB/YCB_Video_Models'

    @classmethod
    def download(cls):
        url = 'https://drive.google.com/uc?id=1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu'
        md5 = 'd3efe74e77fe7d7ca216dde4b7d217fa'

        def postprocess(path):
            contrib.data.extractall(path)
            path_extracted = path.parent / 'models'
            shutil.move(
                str(path_extracted),
                str(cls.root_dir),
            )

        gdown.cached_download(
            url=url,
            path=str(cls.root_dir) + '.zip',
            md5=md5,
            postprocess=postprocess,
        )
