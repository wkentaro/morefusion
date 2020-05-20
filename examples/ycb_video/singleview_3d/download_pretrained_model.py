#!/usr/bin/env python

import gdown
import path


here = path.Path(__file__).abspath().parent


log_dir = here / "logs/logs.20191008.all_data/20191014_092021.638983636"

gdown.cached_download(
    url="https://drive.google.com/uc?id=1uh_B4K1-K4f_Q_uOTA0VnH9z8BBP3avE",
    path=log_dir / "snapshot_model_best_auc.npz",
    md5="65f1530076c39c89b8424073fd42a3c2",
)
gdown.cached_download(
    url="https://drive.google.com/uc?id=1gHA1j577rskXi-8dZUoKexZ-GVp5gczw",
    path=log_dir / "args.json",
    md5="c1f69b56df481bc8c2cb07f94bf114e1",
)
