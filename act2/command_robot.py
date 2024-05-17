import argparse
import os

import rospy

import intera_interface
import intera_external_devices

from intera_interface import CHECK_VERSION
import copy
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from sensor_msgs.msg import Image
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
import os
import h5py



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--command", type=str, default="up",
                      help="the target color to pick up")
  args = parser.parse_args()
  
  command = args.command
  
  if 'sort_target_to_box' in command:
    
    filename = "/home/boxjod/sawyer_ws/intera2.sh"
    with open(filename, "r", encoding='utf-8') as file:
        content = file.readlines()    # 读取文件的所有行
    content[183 - 1] = command + '\n'
    with open(filename, "w", encoding="utf-8") as file:
        file.writelines(content)    # 将更新后的内容写入文件
    os.system("cd /home/boxjod/sawyer_ws ;" + 'bash ' + filename)
     
     

if __name__ == '__main__':
    main()
