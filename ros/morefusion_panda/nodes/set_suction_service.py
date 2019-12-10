#!/usr/bin/env python

import rospy
import serial
from morefusion_panda.srv import SetSuction


ser = serial.Serial('/dev/arduino0', baudrate=9600, timeout=1)


def handle_set_suction(req):

    try:
        if req.on:
            ser.write(b'g')
        else:
            ser.write(b's')
    except Exception:
        return False
    return True


def main():

    rospy.init_node('set_suction_server', anonymous=True)
    rospy.Service('set_suction', SetSuction, handle_set_suction)
    rospy.spin()


if __name__ == '__main__':
    main()
