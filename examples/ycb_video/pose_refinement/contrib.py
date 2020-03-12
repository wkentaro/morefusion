import functools
import re

import imgviz
import numpy as np
import path
import trimesh
import yaml


here = path.Path(__file__).abspath().parent


def grid(scale=1):
    path = []
    N = 10
    for i in range(2):
        for x in range(-N, N + 1):
            if i == 0:
                path.append(trimesh.load_path([[x, -N], [x, N]]))
            else:
                path.append(trimesh.load_path([[-N, x], [N, x]]))
    path = functools.reduce(lambda x, y: x + y, path)
    path.apply_scale(scale)
    return path


def get_data():
    data = {}

    data["instances"] = []
    for npz_file in sorted((here / "data").listdir()):
        if not re.match(r"[0-9]+.npz", npz_file.basename()):
            continue
        instance = dict(np.load(npz_file))
        instance["id"] = int(npz_file.basename().stem)
        data["instances"].append(instance)

    data["rgb"] = imgviz.io.imread("data/image.png")

    depth = np.load("data/depth.npz")["arr_0"]
    depth = depth.astype(np.float32) / 1000
    data["depth"] = depth

    with open(f"data/camera_info.yaml") as f:
        camera_info = yaml.safe_load(f)
    data["intrinsic_matrix"] = np.array(camera_info["K"]).reshape(3, 3)

    return data
