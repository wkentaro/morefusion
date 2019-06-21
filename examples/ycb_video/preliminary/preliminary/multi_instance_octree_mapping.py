import numpy as np
import octomap
import sklearn.neighbors
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
        grid_target = np.zeros(dimensions, dtype=np.float32)
        grid_nontarget = np.zeros(dimensions, dtype=np.float32)
        grid_empty = np.zeros(dimensions, np.float32)

        centers = trimesh.voxel.matrix_to_points(
            np.ones(dimensions), pitch=pitch, origin=origin
        )

        for ins_id, octree in self._octrees.items():
            occupied, empty = octree.extractPointCloud()

            if occupied.size:
                kdtree = sklearn.neighbors.KDTree(occupied)
                dist, indices = kdtree.query(centers, k=1)

                if ins_id == target_id:
                    # occupied by target
                    g = np.minimum(1, np.maximum(0, 1 - (dist / pitch)))
                    g = g.reshape(dimensions)
                    grid_target = np.maximum(grid_target, g)
                else:
                    # occupied by non-target
                    g = np.minimum(1, np.maximum(0, 1 - (dist / pitch)))
                    g = g.reshape(dimensions)
                    grid_nontarget = np.maximum(grid_nontarget, g)

            if empty.size:
                # empty
                kdtree = sklearn.neighbors.KDTree(empty)
                dist, indices = kdtree.query(centers, k=1)
                g = np.minimum(1, np.maximum(0, 1 - (dist / pitch)))
                g = g.reshape(dimensions)
                grid_empty = np.maximum(grid_empty, g)

        return grid_target, grid_nontarget, grid_empty

    def get_target_pcds(self, target_id):
        octree = self._octrees[target_id]
        if target_id not in self._pcds:
            occupied, empty = octree.extractPointCloud()
            self._pcds[target_id] = (occupied, empty)
        return self._pcds[target_id]
