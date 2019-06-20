import chainer
import gdown
import path


def get_eval_result(name):
    root_dir = chainer.dataset.get_dataset_directory('wkentaro/objslampp/ycb_video/dense_fusion/eval_result/ycb')  # NOQA
    root_dir = path.Path(root_dir)

    if name == 'Densefusion_iterative_result':
        zip_file = gdown.cached_download(
            url='https://drive.google.com/uc?id=1sqgpOgcFJB0P4Nx5vwHN8vKj_R0Ng0S0',  # NOQA
            path=root_dir / 'Densefusion_iterative_result.zip',
            postprocess=gdown.extractall,
            md5='46a8b4a16de5b2f87c969c793b1d1825',
        )
    elif name == 'Densefusion_wo_refine_result':
        zip_file = gdown.cached_download(
            url='https://drive.google.com/uc?id=179MtZOvKNR1aJ310GIv0Gi2yf9nb13q3',  # NOQA
            path=root_dir / 'Densefusion_wo_refine_result.zip',
            postprocess=gdown.extractall,
            md5='fbe2524635c44e64af94ab7cf7a19e9d',
        )
    elif name == 'Densefusion_icp_result':
        zip_file = root_dir / 'Densefusion_icp_result.zip'
    else:
        raise ValueError
    result_dir = path.Path(zip_file[:- len('.zip')])

    return result_dir
