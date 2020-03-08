#!/usr/bin/env python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import numpy as np

from morefusion.geometry import points_from_angles


def main():
    points = points_from_angles(
        distance=[1] * 3,
        elevation=[0, 45, 90],
        azimuth=[0] * 3,
        is_degree=True,
    )
    print(points)

    ax = plt.subplot(111, projection="3d")

    # plot unit sphere
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    for point in points:
        # plot point
        xs = [0, point[0]]
        ys = [0, point[1]]
        zs = [0, point[2]]
        ax.plot(xs, ys, zs, marker="o", color="b")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()


if __name__ == "__main__":
    main()
