import numpy as np
import octomap
import trimesh


class MultiInstanceOctreeMapping:

    def __init__(self):
        self._octrees = {}  # key: instance_id, value: octree
        self._pcds = {}     # key: instance_id, value: (occupied, empty)

    @property
    def instance_ids(self):
        return list(self._octrees.keys())

    def initialize(self, instance_id, *, pitch):
        if instance_id in self.instance_ids:
            raise ValueError('instance {instance_id} already exists')
        self._octrees[instance_id] = octomap.OcTree(pitch)

    def integrate(self, instance_id, mask, pcd, origin=(0, 0, 0)):
        origin = np.asarray(origin, dtype=float)
        octree = self._octrees[instance_id]
        nonnan = ~np.isnan(pcd).any(axis=2)
        octree.insertPointCloud(pcd[mask & nonnan], origin=origin)
        if instance_id in self._pcds:
            self._pcds.pop(instance_id)  # clear cache

    def update(self, instance_id, occupied):
        octree = self._octrees[instance_id]
        octree.updateNodes(occupied, True, lazy_eval=True)
        octree.updateInnerOccupancy()
        if instance_id in self._pcds:
            self._pcds.pop(instance_id)  # clear cache

    def get_target_grids(self, target_id, *, dimensions, pitch, origin):
        '''Get voxel grids of the specified instance.

        Parameters
        ----------
        target_id: int
            Target instance ID.
        dimensions: (3,) array-like, int
            Voxel dimensions (e.g., 32x32x32).
        pitch: float
            Voxel pitch.
        origin: (3,) array-like, float
            Voxel origin.

        Returns
        -------
        grid_target: numpy.ndarray
            Occupied space of the target instance.
        grid_nontarget: numpy.ndarray
            Occupied space of non-target instance.
        grid_empty: numpy.ndarray
            Empty space.
        '''
        assert not np.isnan(origin).any()
        assert len(dimensions) == 3
        assert (np.asarray(dimensions) > 0).all()
        assert pitch > 0

        grid_target = np.zeros(dimensions, dtype=np.float32)
        grid_nontarget = np.zeros(dimensions, dtype=np.float32)
        grid_empty = np.zeros(dimensions, np.float32)

        centers = trimesh.voxel.matrix_to_points(
            np.ones(dimensions), pitch=pitch, origin=origin
        )
        indices = trimesh.voxel.points_to_indices(
            centers, pitch=pitch, origin=origin
        )
        I, J, K = indices[:, 0], indices[:, 1], indices[:, 2]

        def get_occupancy(octree, point):
            node = octree.search(point)
            try:
                return node.getOccupancy()
            except octomap.NullPointerException:
                return -1

        for ins_id, octree in self._octrees.items():
            occupancies = np.array([
                get_occupancy(octree, center) for center in centers
            ])
            q = occupancies >= 0.5
            if ins_id == target_id:
                grid_target[I[q], J[q], K[q]] = occupancies[q]
            else:
                grid_nontarget[I[q], J[q], K[q]] = occupancies[q]
            q = (0 <= occupancies) & (occupancies < 0.5)
            grid_empty[I[q], J[q], K[q]] = 1 - occupancies[q]

        return grid_target, grid_nontarget, grid_empty

    def get_target_pcds(self, target_id, aabb_min=None, aabb_max=None):
        '''Get point clouds of the specified instance.

        Parameters
        ----------
        target_id: int
            Target instance ID.
        aabb_min: (3,) array-like, optional
            AABB minimum xyz.
        aabb_max: (3,) array-like, optional
            AABB maximum xyz.

        Returns
        -------
        occupied: (N, 3) numpy.ndarray, np.float64
            Occupied points.
        empty: (M, 3) numpy.ndarray, np.float64
            Empty points.
        '''
        octree = self._octrees[target_id]
        if target_id not in self._pcds:
            occupied, empty = octree.extractPointCloud()
            if aabb_min is not None:
                occupied = occupied[(occupied >= aabb_min).all(axis=1)]
                empty = empty[(empty >= aabb_min).all(axis=1)]
            if aabb_max is not None:
                occupied = occupied[(occupied < aabb_max).all(axis=1)]
                empty = empty[(empty < aabb_max).all(axis=1)]
            self._pcds[target_id] = (occupied, empty)
        return self._pcds[target_id]
