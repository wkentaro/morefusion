#!/usr/bin/env python

import gdown

import morefusion


def main():
    root_dir = morefusion.utils.get_data_path(
        'wkentaro/morefusion/ycb_video/dense_fusion/eval_result/ycb'
    )

    gdown.cached_download(
        url='https://drive.google.com/uc?id=1sqgpOgcFJB0P4Nx5vwHN8vKj_R0Ng0S0',  # NOQA
        path=root_dir / 'Densefusion_iterative_result.zip',
        postprocess=gdown.extractall,
        md5='46a8b4a16de5b2f87c969c793b1d1825',
    )
    gdown.cached_download(
        url='https://drive.google.com/uc?id=179MtZOvKNR1aJ310GIv0Gi2yf9nb13q3',  # NOQA
        path=root_dir / 'Densefusion_wo_refine_result.zip',
        postprocess=gdown.extractall,
        md5='fbe2524635c44e64af94ab7cf7a19e9d',
    )


if __name__ == '__main__':
    main()
