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
        bag_file = '/home/wkentaro/Gdrive/objslampp/ros_objslampp/rs_rgbd_2019-10-21-16-56-28.bag'  # NOQA
    elif bag_id == 'static.desk':
        bag_file = gdown.cached_download(
            url='https://drive.google.com/uc?id=1XPJMTw4VVmUvaMgPXHN0izSWYXRgnhl8',  # NOQA
            md5='4615655dc096d6bad85d15a51a98341b',
        )
    else:
        raise ValueError(f'Unknown bag_id: {bag_id}')
    return bag_file


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--id', choices=['dynamic.desk', 'static.desk', 'static.robot']
    )
    parser.add_argument('--rosbag-args', nargs=argparse.REMAINDER)
    args = parser.parse_args(rospy.myargv()[1:])

    bag_file = get_bag_file(args.id)

    # wait for some nodes launched
    time.sleep(10)

    cmd = f"rosbag play {bag_file} {' '.join(args.rosbag_args)}"
    print(f'+ {cmd}')
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
