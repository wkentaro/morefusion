import chainercv
import imgviz
import numpy as np


class VOCBackgroundComposite:

    def __init__(self, bg_instance_ids):
        self._random_state = np.random.mtrand._rand
        self._voc_dataset = chainercv.datasets.VOCBboxDataset()
        self._bg_instance_ids = bg_instance_ids

    def __call__(self, rgb, instance_label):
        rgb = rgb.copy()

        index = self._random_state.randint(0, len(self._voc_dataset))
        bg = self._voc_dataset.get_example_by_keys(index, [0])[0]
        bg = bg.transpose(1, 2, 0)

        H_fg, W_fg = rgb.shape[:2]
        H_bg, W_bg = bg.shape[:2]

        H = max(H_fg, H_bg)
        W = max(W_fg, W_bg)

        scale = max(H / H_bg, W / W_bg)
        H = int(round(scale * H_bg))
        W = int(round(scale * W_bg))
        bg = imgviz.resize(bg, height=H, width=W, backend='opencv')

        y1 = self._random_state.randint(0, H - H_fg + 1)
        y2 = y1 + H_fg
        x1 = self._random_state.randint(0, W - W_fg + 1)
        x2 = x1 + W_fg
        bg_mask = np.isin(instance_label, self._bg_instance_ids)
        rgb[bg_mask] = bg[y1:y2, x1:x2][bg_mask]

        return rgb
