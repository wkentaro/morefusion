#!/usr/bin/env python

import gdown
import path


here = path.Path(__file__).abspath().parent


log_dir = here / "logs/logs.20191008.all_data/20191014_092228.858011713"

gdown.cached_download(
    url="https://drive.google.com/uc?id=1hTHSv0fenSA-iMprDzDGUJZGpXNTU972",
    path=log_dir / "snapshot_model_best_auc.npz",
    md5="51197e9d4234ddef979f8a741b690456",
)
gdown.cached_download(
    url="https://drive.google.com/uc?id=1dX_FiOfz9iB11UbgD2dZjNoxK3Wq_jAR",
    path=log_dir / "args.json",
    md5="c7e2a59368064b713df5dafd622b98dd",
)
