#!/usr/bin/env python

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn

import objslampp
#
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 16
seaborn.set_palette('muted')

df = pandas.read_csv('./data.00000149.csv', index_col=0)
df['visibility'] = np.clip(df['visibility'], 0, 1)
df_pcd = pandas.read_csv('../singleview_pcd/data.00000149.csv', index_col=0)
df_pcd['visibility'] = np.clip(df['visibility'], 0, 1)
df_noocc = pandas.read_csv('./data.wo_occ.00000149.csv', index_col=0)
df_noocc['visibility'] = np.clip(df['visibility'], 0, 1)
df = pandas.concat([df, df_pcd, df_noocc])

# dfv = df.copy()
# step = 0.1
# for max_visibility in np.arange(1, 10 + 1) * step:
#     min_visibility = max_visibility - step
#     visibility = max_visibility - step / 2
#     dfv.loc[(df.visibility < max_visibility) & (df.visibility >= min_visibility), 'visibility'] = visibility
# df = dfv

methods = ['densefusion', 'morefusion-occ', 'morefusion', 'morefusion+icp', 'morefusion+icc', 'morefusion+icc+icp']

df2 = []
for cls_id in np.unique(df.class_id):
    for method in methods:
        mask = (df.class_id == cls_id) & (df.method == method)
        step = 0.05
        for max_visibility in np.arange(1, 20 + 1) * step:
            visibility = max_visibility - step / 2
            min_visibility = max_visibility - step
            df_cls = df[
                mask
                & (df.visibility <= max_visibility)
                & (df.visibility >= min_visibility)
            ]
            if df_cls.size == 0:
                continue
            add_or_add_s = df_cls['add_or_add_s']
            add_s = df_cls['add_s']
            auc = objslampp.metrics.ycb_video_add_auc(add_or_add_s)
            auc_s = objslampp.metrics.ycb_video_add_auc(add_s)
            assert add_or_add_s.ndim == 1
            acc = (add_or_add_s < 0.02).sum() / add_or_add_s.size
            assert add_s.ndim == 1
            acc_s = (add_s < 0.02).sum() / add_s.size
            df2.append({
                'class_id': cls_id,
                'visibility': visibility,
                'method': method,
                'acc_add_or_add_s': acc,
                'acc_add_s': acc_s,
                'add_or_add_s': add_or_add_s.mean(),
                'add_s': add_s.mean(),
                'auc_add_or_add_s': auc,
                'auc_add_s': auc_s,
            })
df2 = pandas.DataFrame(df2)
df3 = df2.groupby(['visibility', 'method']).mean().reset_index()

ax = seaborn.lineplot(x='visibility', y='auc_add_or_add_s', hue='method', style='method', markers=True, dashes=False, data=df3, hue_order=methods)
ax.set_xlabel('Visibility of Object')
ax.set_ylabel('AUC of ADD/ADD-S')
ax.set_xlim(0, 1)
ax.set_yticks(np.arange(0.9, 1.05, step=0.05))
ax.set_ylim(0.9, 1.01)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])
plt.show()
