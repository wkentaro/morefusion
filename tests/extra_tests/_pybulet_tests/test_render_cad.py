import numpy as np

from objslampp import geometry
from objslampp.datasets import YCBVideoModels
from objslampp.extra import _pybullet as pybullet_module


def test_render_cad():
    dataset = YCBVideoModels()
    visual_file = dataset.get_model(class_id=2)['textured_simple']

    H, W = 256, 256

    # batch_size: 1
    Ts_cad2cam = geometry.look_at((1, 1, 1), (0, 0, 0), up=(0, -1, 0))
    rgbs, depths, masks = pybullet_module.render_cad(
        visual_file,
        Ts_cad2cam,
        fovy=45,
        height=H,
        width=W,
    )
    assert Ts_cad2cam.shape == (4, 4)
    assert rgbs.shape == (H, W, 3)
    assert rgbs.dtype == np.uint8
    assert depths.shape == (H, W)
    assert depths.dtype == np.float32
    assert masks.shape == (H, W)
    assert masks.dtype == bool

    # batch_size: N
    Ts_cad2cam = geometry.look_at((1, 1, 1), (0, 0, 0), up=(0, -1, 0))
    Ts_cad2cam = Ts_cad2cam[None].repeat(2, axis=0)
    rgbs, depths, masks = pybullet_module.render_cad(
        visual_file,
        Ts_cad2cam,
        fovy=45,
        height=H,
        width=W,
    )
    assert Ts_cad2cam.shape == (2, 4, 4)
    assert rgbs.shape == (2, H, W, 3)
    assert rgbs.dtype == np.uint8
    assert depths.shape == (2, H, W)
    assert depths.dtype == np.float32
    assert masks.shape == (2, H, W)
    assert masks.dtype == bool
