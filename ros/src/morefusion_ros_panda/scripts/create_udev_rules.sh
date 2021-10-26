#!/bin/bash

echo ""
echo "This script copies udev rules for grippers and their dependencies"
echo "to /etc/udev/rules.d and /usr/local/sbin"
echo ""

sudo cp $(rospack find morefusion_ros_panda)/udev/90-arduino.rules /etc/udev/rules.d

echo ""
echo "Restarting udev"
echo ""
sudo service udev reload
sudo service udev restart
