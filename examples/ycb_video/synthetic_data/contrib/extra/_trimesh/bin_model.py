import trimesh

import objslampp


def bin_model(extents, thickness):
    xlength, ylength, zlength = extents

    mesh = trimesh.Trimesh()

    wall_xp = trimesh.creation.box((thickness, ylength, zlength))
    wall_xn = wall_xp.copy()
    wall_xp.apply_translation((xlength / 2, 0, 0))
    wall_xn.apply_translation((- xlength / 2, 0, 0))
    wall_xp.visual.face_colors = (1., 0., 0., 0.8)
    wall_xn.visual.face_colors = (1., 0., 0., 0.8)
    mesh += wall_xp
    mesh += wall_xn
    # scene.add_geometry(wall_xp)
    # scene.add_geometry(wall_xn)

    wall_yp = trimesh.creation.box((xlength, thickness, zlength))
    wall_yn = wall_yp.copy()
    wall_yp.apply_translation((0, ylength / 2 - thickness / 2, 0))
    wall_yn.apply_translation((0, - ylength / 2 + thickness / 2, 0))
    wall_yp.visual.face_colors = (0., 1., 0., 0.8)
    wall_yn.visual.face_colors = (0., 1., 0., 0.8)
    mesh += wall_yp
    mesh += wall_yn
    # scene.add_geometry(wall_yp)
    # scene.add_geometry(wall_yn)

    wall_zn = trimesh.creation.box((xlength, ylength, thickness))
    wall_zn.apply_translation((0, 0, - zlength / 2 + thickness / 2))
    wall_zn.visual.face_colors = (0., 0., 1., 0.8)
    mesh += wall_zn
    # scene.add_geometry(wall_zn)

    return mesh


if __name__ == '__main__':
    thickness = 0.01
    xlength = 0.3
    ylength = 0.5
    zlength = 0.2

    mesh = bin_model((xlength, ylength, zlength), thickness)
    trimesh.exchange.export.export_mesh(mesh, '/tmp/bin.obj')

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.axis(0.01))
    scene.add_geometry(mesh)
    objslampp.extra.trimesh.show_with_rotation(scene)
