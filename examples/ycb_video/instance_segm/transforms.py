import abc

import imgaug.augmenters as iaa


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

        for index in self._indices:
            in_data[index] = augmenter.augment_image(in_data[index])
