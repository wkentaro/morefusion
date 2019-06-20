import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np
import trimesh.transformations as tf

import objslampp


class OccupancyGridAlignmentModel(chainer.Link):

    def __init__(self, quaternion_init=None, translation_init=None):
        super().__init__()
        with self.init_scope():
            if quaternion_init is None:
                quaternion_init = np.array([1, 0, 0, 0], dtype=np.float32)
            self.quaternion = chainer.Parameter(
                initializer=quaternion_init
            )
            if translation_init is None:
                translation_init = np.zeros((3,), dtype=np.float32)
            self.translation = chainer.Parameter(
                initializer=translation_init
            )

    def forward(
        self,
        points_source,
        grid_target,
        *,
        pitch,
        origin,
        threshold,
    ):
        transform = objslampp.functions.quaternion_matrix(
            self.quaternion[None]
        )
        transform = objslampp.functions.compose_transform(
            transform[:, :3, :3], self.translation[None]
        )

        points_source = objslampp.functions.transform_points(
            points_source, transform
        )[0]
        grid_source = objslampp.functions.occupancy_grid_3d(
            points_source,
            pitch=pitch,
            origin=origin,
            dimension=grid_target.shape[1:],
            threshold=threshold,
        )

        assert grid_target.dtype == np.float32
        occupied_target = grid_target[0]
        intersection = F.sum(occupied_target * grid_source)
        denominator = F.sum(occupied_target)
        reward = intersection / denominator

        assert grid_target.dtype == np.float32
        unoccupied_target = np.maximum(grid_target[1], grid_target[2])
        unoccupied_target = unoccupied_target.astype(np.float32)
        intersection = F.sum(unoccupied_target * grid_source)
        denominator = F.sum(grid_source)
        penalty = intersection / denominator

        loss = - reward + penalty
        return loss


class OccupancyRegistration:

    def __init__(
        self,
        points_source,
        grid_target,
        *,
        pitch,
        origin,
        threshold,
        transform_init,
        gpu=0,
        alpha=0.1,
    ):
        quaternion_init = tf.quaternion_from_matrix(transform_init)
        quaternion_init = quaternion_init.astype(np.float32)
        translation_init = tf.translation_from_matrix(transform_init)
        translation_init = translation_init.astype(np.float32)

        model = OccupancyGridAlignmentModel(quaternion_init, translation_init)

        self._grid_target_cpu = grid_target

        if gpu >= 0:
            model.to_gpu(gpu)
            points_source = model.xp.asarray(points_source)
            grid_target = model.xp.asarray(grid_target)

        self._points_source = points_source
        self._grid_target = grid_target
        self._pitch = pitch
        self._origin = origin
        self._threshold = threshold

        self._optimizer = chainer.optimizers.Adam(alpha=alpha)
        self._optimizer.setup(model)
        model.translation.update_rule.hyperparam.alpha *= 0.1

    @property
    def _transform(self):
        model = self._optimizer.target
        quaternion = cuda.to_cpu(model.quaternion.array)
        translation = cuda.to_cpu(model.translation.array)
        transform = tf.quaternion_matrix(quaternion)
        transform = objslampp.geometry.compose_transform(
            transform[:3, :3], translation
        )
        return transform

    def register_iterative(self, iteration=None):
        iteration = 100 if iteration is None else iteration

        yield self._transform

        for _ in range(iteration):
            model = self._optimizer.target

            loss = model(
                points_source=self._points_source,
                grid_target=self._grid_target,
                pitch=self._pitch,
                origin=self._origin,
                threshold=self._threshold,
            )
            loss.backward()
            self._optimizer.update()
            model.cleargrads()

            # print(f'[{self._iteration:08d}] {loss}')
            # print(f'quaternion:', model.quaternion.array.tolist())
            # print(f'translation:', model.translation.array.tolist())

            yield self._transform

    def register(self, iteration=None):
        for _ in self.register_iterative(iteration=iteration):
            pass
        return self._transform
