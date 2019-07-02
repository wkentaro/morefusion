#!/usr/bin/env python

import chainer
import tqdm

import contrib


for split in ['train', 'syn', 'val']:
    dataset = contrib.datasets.YCBVideoDataset(
        split, return_occupancy_grids=True
    )
    iterator = chainer.iterators.SerialIterator(
        dataset,
        batch_size=1,
        repeat=False,
        shuffle=False,
    )
    for _ in tqdm.tqdm(iterator, total=len(dataset)):
        pass
