class AsType:

    def __init__(self, indices, dtypes):
        assert len(indices) == len(dtypes)
        self._indices = indices
        self._dtypes = dtypes

    def __call__(self, in_data):
        is_tuple = False
        if isinstance(in_data, tuple):
            is_tuple = True
            in_data = list(in_data)

        for index, dtype in zip(self._indices, self._dtypes):
            in_data[index] = in_data[index].astype(dtype)

        if is_tuple:
            in_data = tuple(in_data)
        return in_data


class HWC2CHW:

    def __init__(self, indices):
        self._indices = indices

    def __call__(self, in_data):
        is_tuple = False
        if isinstance(in_data, tuple):
            is_tuple = True
            in_data = list(in_data)

        for index in self._indices:
            in_data[index] = in_data[index].transpose(2, 0, 1)

        if is_tuple:
            in_data = tuple(in_data)
        return in_data


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
