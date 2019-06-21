#!/usr/bin/env python

import argparse
import collections
import concurrent.futures

import imgviz
import matplotlib.pyplot as plt
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
        '--name',
        choices=contrib.EVAL_RESULTS,
        default='Densefusion_iterative_result',
    )
    args = parser.parse_args()

    result_dir = contrib.get_eval_result(name=args.name)

    adds_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_files = sorted(result_dir.glob('*.mat'))
        assert len(result_files) == 2949, len(result_files)
        for result_file in result_files:
            adds = executor.submit(contrib.get_adds, result_file)
            adds_list.append(adds)

    adds_all = collections.defaultdict(list)
    for future in adds_list:
        adds = future.result()
        for cls_id in adds:
            adds_all[cls_id] += adds[cls_id]

    logs_dir = here / 'logs' / args.name
    logs_dir.mkdir_p()

    data = []
    for cls_id, adds in adds_all.items():
        class_name = objslampp.datasets.ycb_video.class_names[cls_id]

        auc, x, y = objslampp.metrics.ycb_video_add_auc(
            adds[0], max_value=0.1, return_xy=True)
        auc_s, x_s, y_s = objslampp.metrics.ycb_video_add_auc(
            adds[1], max_value=0.1, return_xy=True)

        fig = plt.figure(figsize=(20, 5))

        print('auc (add):', auc)
        plt.subplot(121)
        plt.title(f'{class_name}: ADD (AUC={auc * 100:.2f})')
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
        out_file = logs_dir / f'{class_name}.png'
        print('==> Saved ADD curve plot:', out_file)
        imgviz.io.imsave(out_file, img)

        data.append(dict(
            class_id=cls_id,
            class_name=class_name,
            add_auc=auc,
            add_s_auc=auc_s,
        ))

    df = pandas.DataFrame(data)
    print(df)
    df.to_csv(logs_dir / 'data.csv')


if __name__ == '__main__':
    main()
