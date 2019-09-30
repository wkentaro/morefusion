import abc

import chainercv
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import PIL.Image


class InDataMutatingTransform(abc.ABC):

    @abc.abstractmethod
    def transform(self, in_data):
        raise NotImplementedError

    def __call__(self, in_data):
        is_tuple = False
        if isinstance(in_data, tuple):
            is_tuple = True
            in_data = list(in_data)

        self.transform(in_data)

        if is_tuple:
            in_data = tuple(in_data)
        return in_data


class AsType(InDataMutatingTransform):

    def __init__(self, indices, dtypes):
        assert len(indices) == len(dtypes)
        self._indices = indices
        self._dtypes = dtypes

    def transform(self, in_data):
        for index, dtype in zip(self._indices, self._dtypes):
            in_data[index] = in_data[index].astype(dtype)


class HWC2CHW(InDataMutatingTransform):

    def __init__(self, indices):
        self._indices = indices

    def transform(self, in_data):
        for index in self._indices:
            in_data[index] = in_data[index].transpose(2, 0, 1)


class Dict2Tuple:

    def __init__(self, keys):
        self._keys = keys

    def __call__(self, in_data):
        return tuple([in_data[k] for k in self._keys])


class ClassIds2FGClassIds:

    def __init__(self, indices):
        self._indices = indices

    def __call__(self, in_data):
        for index in self._indices:
            in_data[index] -= 1
        return in_data


class Compose:

    def __init__(self, *transforms):
        self._transforms = transforms

    def __call__(self, in_data):
        for transform in self._transforms:
            in_data = transform(in_data)
        return in_data


class Affine(InDataMutatingTransform):

    def __init__(self, rgb_indices, mask_indices, bbox_indices):
        self._rgb_indices = rgb_indices
        self._mask_indices = mask_indices
        self._bbox_indices = bbox_indices

    def transform(self, in_data):
        augmenter = iaa.AffineCv2(
            translate_percent=(-0.4, 0.4),
            rotate=(-180, 180),
            shear=(-15, 15),
        )

        augmenter = augmenter.to_deterministic()
        shape = None
        for index in self._rgb_indices:
            if shape is None:
                shape = in_data[index].shape[:2]
            assert shape == in_data[index].shape[:2]
            in_data[index] = augmenter.augment_image(in_data[index])
        for index in self._mask_indices:
            masks = in_data[index].astype(np.uint8)
            masks = np.array([augmenter.augment_image(m) for m in masks])
            in_data[index] = masks.astype(bool)
        for index in self._bbox_indices:
            bbox_on_img = imgaug.BoundingBoxesOnImage([
                imgaug.BoundingBox(x1, y1, x2, y2)
                for y1, x1, y2, x2 in in_data[index]
            ], shape)
            bbox_on_img = augmenter.augment_bounding_boxes(bbox_on_img)
            bboxes = []
            for bbox in bbox_on_img.bounding_boxes:
                bboxes.append([bbox.y1, bbox.x1, bbox.y2, bbox.x2])
            bboxes = np.array(bboxes, dtype=in_data[index].dtype)
            in_data[index] = bboxes


class RGBAugmentation(InDataMutatingTransform):

    def __init__(self, indices):
        self._indices = indices

    def transform(self, in_data):
        augmenter = iaa.Sequential([
            iaa.ContrastNormalization(alpha=(0.8, 1.2)),
            iaa.WithColorspace(
                to_colorspace='HSV',
                from_colorspace='RGB',
                children=iaa.Sequential([
                    # SV
                    iaa.WithChannels(
                        (1, 2),
                        iaa.Multiply(mul=(0.8, 1.2), per_channel=True),
                    ),
                    # H
                    iaa.WithChannels(
                        (0,),
                        iaa.Multiply(mul=(0.95, 1.05), per_channel=True),
                    ),
                ]),
            ),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.KeepSizeByResize(children=iaa.Resize((0.25, 1.0))),
        ])

        augmenter = augmenter.to_deterministic()
        for index in self._indices:
            in_data[index] = augmenter.augment_image(in_data[index])


class MaskRCNNTransform(object):

    def __init__(self, min_size, max_size, mean):
        self.min_size = min_size
        self.max_size = max_size
        self.mean = mean

    def __call__(self, in_data):
        img, mask, label, bbox = in_data

        # Scaling and mean subtraction
        img, scale = chainercv.links.model.fpn.misc.scale_img(
            img, self.min_size, self.max_size)
        img -= self.mean
        bbox = bbox * scale
        mask = chainercv.transforms.resize(
            mask.astype(np.float32),
            img.shape[1:],
            interpolation=PIL.Image.NEAREST).astype(np.bool)
        return img, bbox, label, mask
