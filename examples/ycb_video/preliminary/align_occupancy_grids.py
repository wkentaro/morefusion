#!/usr/bin/env python

import chainer
import chainer.functions as F
import numpy as np
import pyglet
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
            if grid_target.dtype == np.float32:
                # iou loss is more robust
                intersection = F.sum(grid_target * grid_source)
                union = F.sum(grid_target)
                # union = F.sum(
                #     grid_target + grid_source - grid_target * grid_source
                # )
                iou = intersection / union
                loss = 1 - iou
            else:
                assert grid_target.dtype == np.int32
                occupied_target = (grid_target == 1).astype(np.float32)
                intersection = F.sum(occupied_target * grid_source)
                union = F.sum(occupied_target)
                iou1 = intersection / union

                occupied_untarget = (
                    np.isin(grid_target, [2, 7])).astype(np.float32)
                intersection = F.sum(occupied_untarget * grid_source)
                union = F.sum(occupied_untarget)
                iou2 = intersection / union
                # iou2 = 0

                loss = - iou1 + iou2

        if return_grid:
            return loss, grid_source
        else:
            return loss


def main():
    nstep = 1000
    connectivity = 2
    dim = 16
    class_id = 8

    models = objslampp.datasets.YCBVideoModels()
    pcd_file = models.get_pcd_model(class_id=class_id)
    points = np.loadtxt(pcd_file, dtype=np.float32)

    np.random.seed(2)
    indices = np.random.permutation(len(points))[:1000]
    points_source = points[indices]

    model = OccupancyGridAlignment()

    optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer.setup(model)
    model.translation.update_rule.hyperparam.alpha *= 0.1

    data = np.load('logs/occupancy_grid.npz')
    grid_target = data['labels']
    pitch = data['pitch']
    origin = data['origin']
    T_cad2cam = data['T_cad2cam']

    transform_to_center = tf.translation_matrix(- (origin + pitch * dim / 2))
    origin = np.array((- pitch * dim / 2, ) * 3)

    transform_true = transform_to_center @ T_cad2cam

    cad_file = models.get_cad_model(class_id=class_id)
    cad = trimesh.load(str(cad_file))
    cad.visual = cad.visual.to_color()

    scenes = align(
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
        transform_true,
    )

    def on_update(dt, scenes, viewer):
        if not viewer.play:
            return
        try:
            scene = next(scenes)
        except StopIteration:
            viewer.on_close()
        viewer.scene.geometry = scene.geometry
        viewer._update_vertex_list()

    viewer = trimesh.viewer.SceneViewer(
        next(scenes), resolution=(640, 480), start_loop=False
    )
    viewer.play = False

    @viewer.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.S:
                viewer.play = not viewer.play

    pyglet.clock.schedule_interval(on_update, 1 / 30, scenes, viewer)
    pyglet.app.run()


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
    transform_true,
):
    scene = trimesh.Scene()

    for iteration in range(-1, nstep):
        if iteration != -1:
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
        else:
            transform_pred = np.eye(4)

        # # geom = trimesh.PointCloud(vertices=points_target)
        # geom = trimesh.Trimesh()
        # for xyz in points_target:
        #     geom_i = trimesh.creation.icosphere(radius=0.005)
        #     geom_i.apply_translation(xyz)
        #     geom += geom_i
        # if 'pcd_target' in scene.geometry:
        #     scene.geometry['pcd_target'] = geom
        # else:
        #     scene.add_geometry(geom, geom_name='pcd_target')

        # grid_target
        voxel = trimesh.voxel.Voxel(
            matrix=grid_target == 1, pitch=pitch, origin=origin)
        geom = voxel.as_boxes((1., 0, 0, 0.7))
        if 'grid_target' in scene.geometry:
            scene.geometry['grid_target'] = geom
        else:
            scene.add_geometry(geom, geom_name='grid_target')
        voxel = trimesh.voxel.Voxel(
            matrix=grid_target == 2, pitch=pitch, origin=origin)
        geom = voxel.as_boxes((0, 1., 0, 0.7))
        if 'grid_target2' in scene.geometry:
            scene.geometry['grid_target2'] = geom
        else:
            scene.add_geometry(geom, geom_name='grid_target2')

        # # grid_source
        # grid = grid_source
        # voxel = trimesh.voxel.Voxel(matrix=grid, pitch=pitch, origin=origin)
        # colors = imgviz.depth2rgb(
        #     grid.reshape(1, -1), min_value=0, max_value=1
        # )
        # colors = colors.reshape(dim, dim, dim, 3)
        # colors = np.concatenate(
        #     (colors, np.full((dim, dim, dim, 1), 127)), axis=3
        # )
        # geom = voxel.as_boxes(colors)
        # if 'grid_source' in scene.geometry:
        #     scene.geometry['grid_source'] = geom
        # else:
        #     scene.add_geometry(geom, geom_name='grid_source')

        # cad_source
        cad_copy = cad.copy()
        cad_copy.apply_transform(transform_pred)
        if 'cad_source' in scene.geometry:
            scene.geometry['cad_source'] = cad_copy
        else:
            scene.add_geometry(cad_copy, geom_name='cad_source')

        # cad_target
        cad_copy = cad.copy()
        cad_copy.visual.vertex_colors[:, 3] = 127
        cad_copy.apply_transform(transform_true)
        if 'cad_target' in scene.geometry:
            scene.geometry['cad_target'] = cad_copy
        else:
            scene.add_geometry(cad_copy, geom_name='cad_target')

        scene.set_camera(angles=[np.deg2rad(180), 0, 0], distance=0.3)

        yield scene


if __name__ == '__main__':
    main()
