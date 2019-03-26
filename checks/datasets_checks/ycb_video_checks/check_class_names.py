#!/usr/bin/env python

import objslampp


def main():
    class_names = objslampp.datasets.ycb_video.class_names
    for class_id, class_name in enumerate(class_names):
        print('{:>2d}: {:s}'.format(class_id, class_name))


if __name__ == '__main__':
    main()
