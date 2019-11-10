#!/usr/bin/env python

import argparse
import collections
import concurrent.futures

import imgviz
import matplotlib.pyplot as plt
import numpy as np
import pandas
import path

import objslampp

import contrib


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--result',
        default='/home/wkentaro/data/datasets/wkentaro/objslampp/ycb_video/dense_fusion/eval_result/ycb/Densefusion_wo_refine_result',  # NOQA
        help='result dir',
    )
    parser.add_argument(
        '--out',
        help='out dir',
        required=True,
    )
    args = parser.parse_args()

    args.result = path.Path(args.result)
    args.out = path.Path(args.out)

    adds_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_files = sorted(args.result.glob('*.mat'))
        assert len(result_files) == 2949, len(result_files)
        for result_file in result_files:
            adds = executor.submit(contrib.get_adds, result_file)
            adds_list.append(adds)

    adds_all = collections.defaultdict(list)
    for future in adds_list:
        adds = future.result()
        for cls_id in adds:
            adds_all[cls_id] += adds[cls_id]

    args.out.makedirs_p()

    data = []
    for cls_id, adds in sorted(adds_all.items()):
        class_name = objslampp.datasets.ycb_video.class_names[cls_id]

        adds = np.array(adds)
        add = adds[:, 0]
        add_s = adds[:, 1]

        if cls_id in objslampp.datasets.ycb_video.class_ids_symmetric:
            add_or_add_s = add_s
        else:
            add_or_add_s = add

        assert add_or_add_s.ndim == 1
        accuracy = (add_or_add_s < 0.02).sum() / add_or_add_s.size

        auc, x, y = objslampp.metrics.ycb_video_add_auc(
            add_or_add_s, max_value=0.1, return_xy=True)
        auc_s, x_s, y_s = objslampp.metrics.ycb_video_add_auc(
            add_s, max_value=0.1, return_xy=True)

        fig = plt.figure(figsize=(20, 5))

        print('auc (add):', auc)
        plt.subplot(121)
        plt.title(f'{class_name}: ADD/ADD-S (AUC={auc * 100:.2f})')
        plt.plot(x, y, color='b')
        plt.xlim(0, 0.1)
        plt.ylim(0, 1)
        plt.xlabel('average distance threshold [m]')
        plt.ylabel('accuracy')
        plt.subplot(122)
        plt.title(f'{class_name}: ADD-S (AUC={auc * 100:.2f})')
        plt.plot(x_s, y_s, color='b')
        plt.xlim(0, 0.1)
        plt.ylim(0, 1)
        plt.xlabel('average distance threshold [m]')
        plt.ylabel('accuracy')

        plt.tight_layout()
        img = imgviz.io.pyplot_fig2arr(fig)
        plt.close()
        out_file = args.out / f'{class_name}.png'
        print('==> Saved ADD curve plot:', out_file)
        imgviz.io.imsave(out_file, img)

        data.append(dict(
            class_id=cls_id,
            class_name=class_name,
            add_or_add_s_2cm=accuracy,
            add_or_add_s=auc,
            add_s=auc_s,
        ))

    df = pandas.DataFrame(data)
    print(df)
    print(df.mean(axis=0))
    df.to_csv(args.out / 'data.csv')


if __name__ == '__main__':
    main()
