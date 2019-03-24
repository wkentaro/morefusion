import numpy as np
import trimesh

from objslampp.datasets import YCBVideoModelsDataset
from objslampp.extra import _pybullet as pybullet_module


def test_render_views():
    dataset = YCBVideoModelsDataset()
    visual_file = dataset.get_model(class_id=2)['textured_simple']

    eyes = [(1, 1, 1)]
    targets = [(0, 0, 0)]
    H, W = 256, 256
    K, Ts_cam2world, rgbs, depths, segms = pybullet_module.render_views(
        visual_file,
        eyes,
        targets,
        height=H,
        width=W,
        gui=False
    )

    assert Ts_cam2world.shape == (1, 4, 4)
    assert rgbs.shape == (1, H, W, 3)
    assert depths.shape == (1, H, W)
    assert segms.shape == (1, H, W)

    fovx = 60.
    fovy = fovx / W * H
    K_expected = trimesh.scene.Camera(resolution=(W, H), fov=(fovx, fovy)).K
    np.testing.assert_allclose(K, K_expected)
