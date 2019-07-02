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
        grid_target = np.zeros(dimensions, dtype=np.float32)
        grid_nontarget = np.zeros(dimensions, dtype=np.float32)
        grid_empty = np.zeros(dimensions, np.float32)

        centers = trimesh.voxel.matrix_to_points(
            np.ones(dimensions), pitch=pitch, origin=origin
        )

        for ins_id, octree in self._octrees.items():
            for center in centers:
                node = octree.search(center)
                try:
                    occupancy = node.getValue()
                except octomap.NullPointerException:
                    continue
                i, j, k = trimesh.voxel.points_to_indices(
                    [center], pitch=pitch, origin=origin)[0]
                if occupancy > 0.5:
                    if ins_id == target_id:
                        grid_target[i, j, k] = occupancy
                    else:
                        grid_nontarget[i, j, k] = occupancy
                else:
                    grid_empty[i, j, k] = 1 - occupancy

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
