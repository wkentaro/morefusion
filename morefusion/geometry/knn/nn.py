import math

import numpy as np
import path
import sklearn.neighbors


here = path.Path(__file__).abspath().parent
cu_file = here / 'cuComputeDistanceGlobal.cu'


def nn_cpu(ref, query):
    kdtree = sklearn.neighbors.KDTree(ref)
    indices = kdtree.query(query, return_distance=False)
    return indices[:, 0]


def nn_gpu(ref, query):
    import cupy

    with open(cu_file) as f:
        kernel = cupy.RawKernel(f.read(), 'cuComputeDistanceGlobal')

    ref_nb, ref_dim = ref.shape
    query_nb, query_dim = query.shape
    assert ref_dim == query_dim
    dim = ref_dim

    ref = ref.transpose(1, 0)
    query = query.transpose(1, 0)
    ref = cupy.ascontiguousarray(ref)
    query = cupy.ascontiguousarray(query)

    dist = cupy.empty((ref_nb, query_nb), dtype=cupy.float32)

    BLOCK_DIM = 16
    grid = (
        int(math.ceil(query_nb / BLOCK_DIM)),
        int(math.ceil(ref_nb / BLOCK_DIM)),
        1,
    )
    block = (16, 16, 1)
    args = (ref, ref_nb, query, query_nb, dim, dist)
    shared_mem = BLOCK_DIM * BLOCK_DIM + BLOCK_DIM * BLOCK_DIM + 5

    kernel(grid, block, args=args, shared_mem=shared_mem)

    indices = cupy.argmin(dist, axis=0)
    return indices


def nn(ref, query):
    if isinstance(ref, np.ndarray) and isinstance(query, np.ndarray):
        return nn_cpu(ref, query)
    return nn_gpu(ref, query)


if __name__ == '__main__':
    import cupy

    def nn_naive(ref, query):
        dist = ((ref[None, :, :] - query[:, None, :]) ** 2).sum(axis=2)
        # indices of ref for each query point
        indices = cupy.argmin(dist, axis=1)
        return indices

    ref = cupy.random.random((500, 3), dtype=cupy.float32)
    query = cupy.random.random((500000, 3), dtype=cupy.float32)
    indices1 = nn(ref, query)
    indices2 = nn_naive(ref, query)
    assert (indices1 == indices2).all()
