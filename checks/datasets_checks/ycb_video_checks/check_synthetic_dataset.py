#!/usr/bin/env python

import morefusion

from check_dataset import MainApp as _MainApp


class MainApp(_MainApp):
    def __init__(self):
        self._dataset = morefusion.datasets.YCBVideoSyntheticDataset()
        self._mainloop()


if __name__ == "__main__":
    MainApp()
