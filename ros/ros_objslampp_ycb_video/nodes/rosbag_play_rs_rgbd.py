#!/usr/bin/env python

import argparse
import subprocess
import time

import gdown

import rospy


def get_bag_file(bag_id):
    if bag_id == 'dynamic.desk':
        bag_file = gdown.cached_download(
            url='https://drive.google.com/uc?id=1vaPyJERDNFY7W8VUBT3JvPNXHpZ6Bqun',  # NOQA
            md5='5bfb6eb7f80773dd2b8a6b16a93823e9',
        )
    elif bag_id == 'static.robot':
        bag_file = gdown.cached_download(
            url='https://drive.google.com/uc?id=1DwSQLhrtciKCO566R2TqdPc81QOi7tlN',  # NOQA
            md5='ec155c5eacab3d90e72d7b1286c7aff3',
        )
    elif bag_id == 'static.desk':
        bag_file = gdown.cached_download(
            url='https://drive.google.com/uc?id=1mArVTWl2f0Uws_mRoCtDzttV2s6AahZa',  # NOQA
            md5='a70a792577447a414c8ac7fe5f4aa316',
        )
    elif bag_id == 'static.desk.topdown':
        bag_file = gdown.cached_download(
            url='https://drive.google.com/uc?id=1UxUg4IozQvQNrCALXkzk23v3fBjpFCqZ',  # NOQA
            md5='3625c11cd130a06557f38b8ff390882e',
        )
    else:
        raise ValueError(f'Unknown bag_id: {bag_id}')
    return bag_file


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--id',
        choices=[
            'dynamic.desk',
            'static.robot',
            'static.desk',
            'static.desk.topdown',
        ],
        help='id'
    )
    parser.add_argument(
        '--sleep',
        type=float,
        default=10,
        help='sleep [sec]',
    )
    parser.add_argument('--rosbag-args', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(rospy.myargv()[1:])

    bag_file = get_bag_file(args.id)

    # wait for some nodes launched
    time.sleep(args.sleep)

    cmd = f"rosbag play {bag_file} {' '.join(args.rosbag_args)} "\
          "/tf_static_republished:=/tf_static"
    print(f'+ {cmd}')
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
