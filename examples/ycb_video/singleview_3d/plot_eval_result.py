#!/usr/bin/env python
# flake8: noqa

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn

import objslampp

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('case')
args = parser.parse_args()
#
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 16
seaborn.set_palette('coolwarm_r')
# seaborn.set_palette('Paired')

index = 2999

df_occ = pandas.read_csv(f'./logs.csv_files/data.{index:08d}.csv', index_col=0)
df_occ['visibility'] = np.clip(df_occ['visibility'], 0, 1)
df_pcd = pandas.read_csv(f'../singleview_pcd/logs.csv_files/data.{index:08d}.csv', index_col=0)
df_pcd['visibility'] = np.clip(df_pcd['visibility'], 0, 1)
df_noocc = pandas.read_csv(f'./logs.csv_files/data.wo_occ.{index:08d}.csv', index_col=0)
df_noocc['visibility'] = np.clip(df_noocc['visibility'], 0, 1)
df = pandas.concat([df_occ, df_pcd, df_noocc])

df.loc[df['method'] == 'morefusion-occ', 'method'] = 'morefusion-other_depth'
df.loc[df['method'] == 'morefusion', 'method'] = 'No Refinement'
df.loc[df['method'] == 'morefusion+icp', 'method'] = '+ICP'
df.loc[df['method'] == 'morefusion+icc', 'method'] = '+ICC'
df.loc[df['method'] == 'morefusion+icc+icp', 'method'] = '+ICC+ICP'

case = args.case
if case == 'pred':
    methods = ['densefusion', 'morefusion-other_depth', 'morefusion']
else:
    assert case == 'refine'
    methods = ['No Refinement', '+ICP', '+ICC', '+ICC+ICP']

# step = 0.2
# for max_visibility in np.arange(1, 5 + 1) * step:
#     min_visibility = max_visibility - step
#     visibility = max_visibility - step / 2
#     df.loc[(df.visibility < max_visibility) & (df.visibility >= min_visibility), 'visibility'] = visibility
# df['visibility'] = [f'{x:.1f}' for x in df['visibility']]
# seaborn.violinplot(x='visibility', y='add_or_add_s', hue='method', data=df, hue_order=methods)
# plt.show()
# quit()

df2 = []
for cls_id in np.unique(df.class_id):
    for method in methods:
        mask = (df.class_id == cls_id) & (df.method == method)
        step = 0.2
        for visibility in np.arange(1, 5 + 1) * step:
            min_visibility = visibility - step / 2
            max_visibility = visibility + step / 2
            df_cls = df[
                mask
                & (df.visibility >= min_visibility)
                & (df.visibility <= max_visibility)
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
                'visibility': f'{visibility:.1f}',
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

# ax = seaborn.lineplot(x='visibility', y='auc_add_or_add_s', hue='method', style='method', markers=True, dashes=False, data=df3, hue_order=methods)
# ax = seaborn.lineplot(x='visibility', y='auc_add_s', hue='method', style='method', markers=True, dashes=False, data=df3, hue_order=methods)
ax = seaborn.barplot(x='visibility', y='auc_add_or_add_s', hue='method', data=df3, hue_order=methods)
# ax = seaborn.lineplot(x='visibility', y='auc_add_or_add_s', hue='method', style='method', markers=True, dashes=False, data=df3, hue_order=methods)
ax.set_xlabel('Visibility of Object')
ax.set_ylabel('AUC of ADD(-S)')
# ax.set_xlim(0, 1)
# ax.set_xticks(np.arange(0.1, 1.05, step=0.1))
ax.set_yticks(np.arange(0.5, 1.05, step=0.1))
ax.set_ylim(0.75, 1.0)
plt.gca().legend().set_title('')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[1:], labels=labels[1:])
# ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
