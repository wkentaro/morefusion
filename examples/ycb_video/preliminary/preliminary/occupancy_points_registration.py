import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np
import trimesh.transformations as tf

import objslampp


class OccupancyPointsRegistrationLink(chainer.Link):

    # quaternion_init, translation_init: T_cam2cad
    def __init__(self, quaternion_init=None, translation_init=None):
        super().__init__()

        if quaternion_init is None:
            quaternion_init = np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            assert quaternion_init.shape == (4,)
            assert quaternion_init.dtype == np.float32
        if translation_init is None:
            translation_init = np.array([0, 0, 0], dtype=np.float32)
        else:
            assert translation_init.shape == (3,)
            assert translation_init.dtype == np.float32

        with self.init_scope():
            self.quaternion = chainer.Parameter(initializer=quaternion_init)
            self.translation = chainer.Parameter(initializer=translation_init)

    @property
    def T(self):
        # source -> target
        R = objslampp.functions.quaternion_matrix(self.quaternion[None])[0]
        T = objslampp.functions.translation_matrix(self.translation[None])[0]
        return R @ T

    def forward(
        self,
        pcd_depth_target,
        pcd_depth_nontarget,
        pcd_cad,
        threshold_nontarget=0.02,
    ):
        # source is transformed
        # source is the starting point for nearest neighbor

        loss = 0
        for i in [0, 1]:
            if i == 0:
                source = pcd_depth_target
                target = pcd_cad
            else:
                source = pcd_depth_nontarget
                target = pcd_cad

            source = objslampp.functions.transform_points(
                source, self.T[None])[0]

            dists = F.sum(
                (source[None, :, :] - target[:, None, :]) ** 2, axis=2
            ).array
            correspondence = F.argmin(dists, axis=0).array
            dists = dists[correspondence, np.arange(dists.shape[1])]

            if i == 0:
                keep = dists < 0.02
                source_match = source[keep]
                correspondence = correspondence[keep]
                target_match = target[correspondence]

                dists_match = F.sum((source_match - target_match) ** 2, axis=1)
                loss_i = F.mean(dists_match, axis=0) / 0.02
                loss += loss_i
            elif threshold_nontarget > 0:
                keep = dists < threshold_nontarget
                source_match = source[keep]
                correspondence = correspondence[keep]
                target_match = target[correspondence]

                dists_match = F.sum((source_match - target_match) ** 2, axis=1)
                loss_i = F.mean(0.2 - dists_match) / 0.2
                loss += 0.1 * loss_i
        return loss


class OccupancyPointsRegistration:

    def __init__(
        self,
        pcd_depth_target,
        pcd_depth_nontarget,
        pcd_cad,
        *,
        transform_init,  # T_cad2cam
        gpu=0,
        alpha=0.1,
    ):
        translation_init = - tf.translation_from_matrix(transform_init)
        translation_init = translation_init.astype(np.float32)
        rotation_matrix = objslampp.geometry.compose_transform(
            R=transform_init[:3, :3].T)
        quaternion_init = tf.quaternion_from_matrix(rotation_matrix)
        quaternion_init = quaternion_init.astype(np.float32)

        link = OccupancyPointsRegistrationLink(
            quaternion_init, translation_init
        )

        if gpu >= 0:
            link.to_gpu(gpu)
            pcd_depth_target = link.xp.asarray(pcd_depth_target)
            pcd_depth_nontarget = link.xp.asarray(pcd_depth_nontarget)
            pcd_cad = link.xp.asarray(pcd_cad)

        self._pcd_depth_target = pcd_depth_target
        self._pcd_depth_nontarget = pcd_depth_nontarget
        self._pcd_cad = pcd_cad

        self._optimizer = chainer.optimizers.Adam(alpha=alpha)
        self._optimizer.setup(link)
        link.translation.update_rule.hyperparam.alpha *= 0.1

    @property
    def _transform(self):
        link = self._optimizer.target
        T_cam2cad = link.T.array
        return np.linalg.inv(cuda.to_cpu(T_cam2cad))

    def register_iterative(self, iteration=None):
        iteration = 100 if iteration is None else iteration

        yield self._transform

        for i in range(iteration):
            link = self._optimizer.target

            loss = link(
                pcd_depth_target=self._pcd_depth_target,
                pcd_depth_nontarget=self._pcd_depth_nontarget,
                pcd_cad=self._pcd_cad,
                threshold_nontarget=0.02 * (iteration - i) / iteration,
            )
            loss.backward()
            self._optimizer.update()
            link.cleargrads()

            # print(f'[{self._iteration:08d}] {loss}')
            # print(f'quaternion:', model.quaternion.array.tolist())
            # print(f'translation:', model.translation.array.tolist())

            yield self._transform

    def register(self, iteration=None):
        for _ in self.register_iterative(iteration=iteration):
            pass
        return self._transform
