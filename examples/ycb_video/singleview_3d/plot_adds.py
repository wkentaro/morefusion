#!/usr/bin/env python

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas

import morefusion


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--metric', default='add', help='add or add_s')
args = parser.parse_args()

metric = args.metric
models = morefusion.datasets.YCBVideoModels()

# w/o occupancy
with open('./logs/20190710_180413/log') as f:
    data = json.load(f)
df_wo = pandas.DataFrame(data)

# w/ occupancy
with open('./logs/20190711_110540/log') as f:
    data = json.load(f)
df_wi = pandas.DataFrame(data)

columns = df_wo.columns
assert np.all(columns == df_wi.columns)

fig, axes = plt.subplots(6, 6)

ymin = float('inf')
ymax = - float('inf')
index = 0
axes_plot = []
for class_id in range(0, 1 + axes.size):
    if class_id == 0:
        column = f'validation/main/{metric}'
        class_id = None
    else:
        column = f'validation/main/{metric}/{class_id:04d}'
        if column not in columns:
            continue

    row = 2 * (index // axes.shape[1])
    col = index % axes.shape[1]

    ax = axes[row][col]
    df_wo[column].dropna().plot(
        label=f'w/o occ ({class_id})',
        ax=ax,
    )
    df_wi[column].dropna().plot(
        label=f'w/ occ ({class_id})',
        ax=ax,
    )
    ax.set_xlabel('iteration')
    ax.set_ylabel(f'{metric}')
    ax.legend()
    axes_plot.append(ax)

    # -------------------------------------------------------------------------

    ax = axes[row + 1][col]
    if class_id is not None:
        cad_file = models.get_cad_file(class_id=class_id)
        top_image = morefusion.extra.pybullet.get_top_image(cad_file)
        ax.imshow(top_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # -------------------------------------------------------------------------

    ymin = min(ymin, df_wo[column].min(), df_wi[column].min())
    ymax = max(ymax, df_wo[column].max(), df_wi[column].max())

    index += 1

while True:
    row = 2 * (index // axes.shape[1])
    col = index % axes.shape[1]
    try:
        ax = axes[row][col]
    except IndexError:
        break
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = axes[row + 1][col]
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    index += 1

for ax in axes_plot:
    ax.set_ylim(ymin, ymax)

plt.show()
