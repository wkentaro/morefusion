#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn

import objslampp


seaborn.set()

df = pandas.read_csv('data.csv', index_col=0)
df['visibility'] = np.clip(df['visibility'], 0, 1)

# dfv = df.copy()
# step = 0.1
# for max_visibility in np.arange(1, 10 + 1) * step:
#     min_visibility = max_visibility - step
#     visibility = max_visibility - step / 2
#     dfv.loc[(df.visibility < max_visibility) & (df.visibility >= min_visibility), 'visibility'] = visibility
# df = dfv

df2 = []
for cls_id in np.unique(df.class_id):
    for method in np.unique(df.method):
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
            df2.append({
                'class_id': cls_id,
                'visibility': visibility,
                'method': method,
                'add_or_add_s': add_or_add_s.mean(),
                'add_s': add_s.mean(),
                'auc_add_or_add_s': auc,
                'auc_add_s': auc_s,
            })
df2 = pandas.DataFrame(df2)
df3 = df2.groupby(['visibility', 'method']).mean().reset_index()
hue_order = ['+occupancy', '+occupancy+icp',
             '+occupancy+icc', '+occupancy+icc+icp']
# plt.subplot(221)
# ax = seaborn.lineplot(x='visibility', y='auc_add_or_add_s', hue='method', style='method', markers=True, dashes=False, data=df3, hue_order=hue_order,)
# ax.set_xlim(0, 1)
# plt.subplot(222)
ax = seaborn.lineplot(x='visibility', y='auc_add_s', hue='method', style='method',
                      markers=True, dashes=False, data=df3, hue_order=hue_order,)
# plt.axvline(0.175, linestyle='dotted', color='k')
# x_ticks = np.append(ax.get_xticks(), 0.175)
# ax.set_xticks(x_ticks)
ax.set_xlim(0, 1)
# plt.subplot(223)
# ax = seaborn.lineplot(x='visibility', y='add_or_add_s', hue='method', style='method', markers=True, dashes=False, data=df3, hue_order=hue_order,)
# ax.set_xlim(0, 1)
# plt.subplot(224)
# ax = seaborn.lineplot(x='visibility', y='add_s', hue='method', style='method', markers=True, dashes=False, data=df3, hue_order=hue_order,)
# ax.set_xlim(0, 1)
plt.show()
