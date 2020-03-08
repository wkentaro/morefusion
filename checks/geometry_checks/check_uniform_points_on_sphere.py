#!/usr/bin/env python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import numpy as np

from morefusion.geometry import uniform_points_on_sphere


def main():
    points = uniform_points_on_sphere(angle_sampling=10)

    ax = plt.subplot(111, projection="3d")

    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    xs, ys, zs = zip(*points)
    ax.plot(xs, ys, zs, marker="o", linestyle="None", color="b")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()


if __name__ == "__main__":
    main()
