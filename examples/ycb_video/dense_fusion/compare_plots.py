#!/usr/bin/env python

import path
import pandas

import contrib


here = path.Path(__file__).abspath().parent

symmetric_objects = [
    '024_bowl',
    '036_wood_block',
    '051_large_clamp',
    '052_extra_large_clamp',
    '061_foam_brick',
]

dfs = []
for name in contrib.EVAL_RESULTS:
    csv_file = here / f'logs/{name}/data.csv'
    if not csv_file.exists():
        continue
    df = pandas.read_csv(csv_file, index_col=0)
    df = df[~df['class_name'].isin(symmetric_objects)]
    df = df[['class_id', 'class_name', 'add_auc']]
    df = df.set_index(['class_id', 'class_name'])
    df = df.rename(columns={'add_auc': f'{name} (ADD)'})
    dfs.append(df)
df = pandas.concat(dfs, axis=1, sort=False)
print(df)
