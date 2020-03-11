import chainer
import chainer.functions as F
import numpy as np
import trimesh.transformations as ttf

import morefusion


class IterativeClosestPointLink(chainer.Link):
    def __init__(self, transform):
        super().__init__()

        quaternion = ttf.quaternion_from_matrix(transform).astype(np.float32)
        translation = ttf.translation_from_matrix(transform).astype(np.float32)

        with self.init_scope():
            self.quaternion = chainer.Parameter(initializer=quaternion)
            self.translation = chainer.Parameter(initializer=translation)

    @property
    def T(self):
        return morefusion.functions.transformation_matrix(
            self.quaternion, self.translation
        )

    def forward(self, source, target):
        # source: from cad
        # target: from depth

        source = morefusion.functions.transform_points(source, self.T[None])[0]

        dists = F.sum(
            (source[None, :, :] - target[:, None, :]) ** 2, axis=2
        ).array
        correspondence = F.argmin(dists, axis=1).array
        dists = dists[np.arange(dists.shape[0]), correspondence]

        keep = dists < 0.02
        target_match = target[keep]
        correspondence = correspondence[keep]
        source_match = source[correspondence]

        loss = F.sum(F.sum((source_match - target_match) ** 2, axis=1), axis=0)
        return loss
