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

for use_add_s in [0, 1]:
    dfs = []
    for name in contrib.EVAL_RESULTS:
        csv_file = here / f'logs/{name}/data.csv'
        if not csv_file.exists():
            continue
        df = pandas.read_csv(csv_file, index_col=0)
        df = df[~df['class_name'].isin(symmetric_objects)]
        if use_add_s:
            df = df[['class_id', 'class_name', 'add_s_auc']]
        else:
            df = df[['class_id', 'class_name', 'add_auc']]
        df = df.set_index(['class_id', 'class_name'])
        if use_add_s:
            df = df.rename(columns={'add_s_auc': f'{name} (ADD-S)'})
        else:
            df = df.rename(columns={'add_auc': f'{name} (ADD)'})
        dfs.append(df)
    df = pandas.concat(dfs, axis=1, sort=False)
    df = df.sort_index()
    # print(df)

    if use_add_s:
        baseline_name = 'Densefusion_wo_refine_result (ADD-S)'
    else:
        baseline_name = 'Densefusion_wo_refine_result (ADD)'
    baseline = df[baseline_name]
    for col in df.columns:
        if col != baseline_name:
            df[col] -= baseline
    print(df)

    print(df.mean())
