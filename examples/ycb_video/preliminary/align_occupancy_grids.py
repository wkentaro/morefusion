#!/usr/bin/env python

import argparse

import chainer
import chainer.functions as F
import imgviz
import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tf

import objslampp


class OccupancyGridAlignment(chainer.Link):

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
        grid_target,
        points_source,
        *,
        pitch,
        origin,
        connectivity,
        return_grid=False,
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
            dimension=grid_target.shape,
            connectivity=connectivity,
        )

        if 0:
            mask = grid_target > 0
            loss1 = F.mean_squared_error(grid_target[mask], grid_source[mask])
            mask = grid_source.array > 0
            loss2 = F.mean_squared_error(grid_target[mask], grid_source[mask])
            loss = loss1 + loss2
            # loss = F.mean_squared_error(grid_target, grid_source)
        else:
            # iou loss is more robust
            intersection = F.sum(grid_target * grid_source)
            union = F.sum(
                grid_target + grid_source - grid_target * grid_source
            )
            iou = intersection / union
            loss = 1 - iou

        if return_grid:
            return loss, grid_source
        else:
            return loss


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--case', type=int, default=0, choices=[0, 1, 2], help='case'
    )
    args = parser.parse_args()

    nstep = 40
    connectivity = 2
    dim = 8

    if args.case == 0:
        class_id = 2
        R = tf.rotation_matrix(np.deg2rad(180), [1, 0, 0])
        quaternion_init = tf.quaternion_from_matrix(R).astype(np.float32)
        translation_init = np.zeros((3,), dtype=np.float32)
    elif args.case == 1:
        class_id = 20
        np.random.seed(4)
        quaternion_init = tf.random_quaternion().astype(np.float32)
        translation_init = np.random.uniform(
            -0.1, 0.1, (3,)
        ).astype(np.float32)
    elif args.case == 2:
        class_id = 20
        R = tf.rotation_matrix(np.deg2rad(90), [1, 0, 0])
        R = tf.rotation_matrix(np.deg2rad(180), [0, 1, 0]) @ R
        quaternion_init = tf.quaternion_from_matrix(R).astype(np.float32)
        translation_init = np.zeros((3,), dtype=np.float32)

    models = objslampp.datasets.YCBVideoModels()
    pcd_file = models.get_pcd_model(class_id=class_id)
    points = np.loadtxt(pcd_file, dtype=np.float32)

    indices = np.random.permutation(len(points))[:1000]
    points_target = points[indices]
    indices = np.random.permutation(len(points))[:500]
    points_source = points[indices]

    model = OccupancyGridAlignment(
        quaternion_init=quaternion_init,
        translation_init=translation_init,
    )

    if 1:
        optimizer = chainer.optimizers.Adam(alpha=0.1)
        optimizer.setup(model)
        model.translation.update_rule.hyperparam.alpha *= 0.1
    else:
        optimizer = chainer.optimizers.MomentumSGD(lr=0.01)
        optimizer.setup(model)
        model.translation.update_rule.hyperparam.lr *= 0.01

    pitch = max(
        points_target.max(axis=0) - points_target.min(axis=0)
    ) * 1.1 / dim
    origin = (- pitch * dim / 2,) * 3
    dimension = (dim,) * 3
    grid_target = objslampp.functions.occupancy_grid_3d(
        points_target,
        pitch=pitch,
        origin=origin,
        dimension=dimension,
        connectivity=connectivity,
    ).array
    grid_target[grid_target > 0] = 1

    cad_file = models.get_cad_model(class_id=class_id)
    cad = trimesh.load(str(cad_file))
    cad.visual = cad.visual.to_color()

    eyes = objslampp.geometry.uniform_points_on_sphere(angle_sampling=3)
    transforms_init = [
        objslampp.geometry.look_at(eye, target=(0, 0, 0)) for eye in eyes
    ]
    for t in transforms_init:
        t[:3, 3] = 0
    points_source_org = points_source
    for transform_init in transforms_init:
        points_source = tf.transform_points(points_source_org, transform_init)
        points_source = points_source.astype(np.float32)

        align(
            model,
            grid_target,
            points_source,
            optimizer,
            connectivity,
            nstep,
            pitch,
            origin,
            dim,
            cad,
            transform_init,
        )


def align(
    model,
    grid_target,
    points_source,
    optimizer,
    connectivity,
    nstep,
    pitch,
    origin,
    dim,
    cad,
    transform_init,
):
    play = 0
    for iteration in range(nstep):
        loss, grid_source = model(
            grid_target,
            points_source,
            pitch=pitch,
            origin=origin,
            connectivity=connectivity,
            return_grid=True,
        )
        grid_source = grid_source.array
        loss.backward()
        optimizer.update()
        model.cleargrads()

        loss = float(loss.array)
        print(f'[{iteration:08d}] {loss}')
        print(f'quaternion:', model.quaternion.array.tolist())
        print(f'translation:', model.translation.array.tolist())

        quaternion_pred = model.quaternion.array
        translation_pred = model.translation.array
        transform_pred = tf.quaternion_matrix(quaternion_pred)
        transform_pred = objslampp.geometry.compose_transform(
            transform_pred[:3, :3], translation_pred
        )
        transform_pred = transform_pred @ transform_init

        scene_grid = pyrender.Scene()
        # grid_target
        grid = grid_target
        voxel = trimesh.voxel.Voxel(matrix=grid, pitch=pitch, origin=origin)
        geom = voxel.as_boxes()
        geom.apply_translation((pitch / 2, pitch / 2, pitch / 2))
        I, J, K = zip(*np.argwhere(grid))
        colors = imgviz.depth2rgb(
            grid.reshape(1, -1), min_value=0, max_value=1
        )
        colors = colors.reshape(dim, dim, dim, 3)
        colors = np.concatenate(
            (colors, np.full((dim, dim, dim, 1), 127)), axis=3
        )
        geom.visual.face_colors = colors[I, J, K].repeat(12, axis=0)
        geom = pyrender.Mesh.from_trimesh(geom, smooth=False)
        scene_grid.add(geom)
        # grid_source
        grid = grid_source
        voxel = trimesh.voxel.Voxel(matrix=grid, pitch=pitch, origin=origin)
        geom = voxel.as_boxes()
        geom.apply_translation((pitch / 2, pitch / 2, pitch / 2))
        I, J, K = zip(*np.argwhere(grid))
        colors = imgviz.depth2rgb(
            grid.reshape(1, -1), min_value=0, max_value=1
        )
        colors = colors.reshape(dim, dim, dim, 3)
        colors = np.concatenate(
            (colors, np.full((dim, dim, dim, 1), 127)), axis=3
        )
        geom.visual.face_colors = colors[I, J, K].repeat(12, axis=0)
        geom = pyrender.Mesh.from_trimesh(geom, smooth=False)
        scene_grid.add(geom)

        scene = pyrender.Scene()
        # cad_true
        cad_true = cad.copy()
        cad_true.visual.vertex_colors[:, 3] = 200
        scene.add(pyrender.Mesh.from_trimesh(cad_true, smooth=False))
        # cad_pred
        cad_pred = cad.copy()
        scene.add(
            pyrender.Mesh.from_trimesh(cad_pred, smooth=False),
            pose=transform_pred,
        )

        # position
        eye = objslampp.geometry.points_from_angles(
            distance=0.4, elevation=30, azimuth=45
        )
        cam_pose = objslampp.geometry.look_at(eye)
        cam_pose = objslampp.extra.trimesh.camera_transform(cam_pose)
        # camera
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45))
        scene.add(camera, pose=cam_pose)
        scene_grid.add(camera, pose=cam_pose)
        # spotlight
        spotlight = pyrender.SpotLight(intensity=3)
        scene.add(spotlight, pose=cam_pose)
        scene_grid.add(spotlight, pose=cam_pose)
        # rendering
        renderer = pyrender.OffscreenRenderer(640, 480)
        viz_cad = renderer.render(scene)[0][:, :, :3]
        viz_grid = renderer.render(scene_grid)[0][:, :, :3]
        renderer.delete()

        viz_cad = imgviz.draw.text_in_rectangle(
            viz_cad,
            loc='lt',
            text=f'[{iteration:04d}]: dim={dim:02d}, loss={loss:2g}',
            size=16,
            background=(0, 255, 0),
        )
        viz = imgviz.tile([viz_cad, viz_grid], (1, 2), border=(0, 0, 0))
        # imgviz.io.imsave(f'logs/{iteration:04d}.jpg', viz)
        imgviz.io.cv_imshow(viz)
        key = imgviz.io.cv_waitkey(play)
        if key == ord('q'):
            break
        elif key == ord('s'):
            play = not play


if __name__ == '__main__':
    main()
