#!/usr/bin/env python
## -*- coding: utf-8 -*-
import roslib
import sys
import cv2
import rospy
import time
import pypinyin
import os
import numpy as np
from std_msgs.msg import String
import cutpig
import text_re
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from cv_bridge import CvBridge, CvBridgeError
from visual_grab.msg import ocr_posemsg

#os.system("python demoa.py")
sw = 2

img = [] 

#回调函数将ros图像转为cv图像
def callback(data):
    global img
    
    try:
       image = CvBridge().imgmsg_to_cv2(data, "bgr8")
       #cv2.imshow('image', image)
       #cv2.waitKey(10)
       img = image
       return img
    except CvBridgeError as e:
       print(e)



def ocrcallback(data):
    global sw
    switch = data.data
    
    sw = switch
    return sw


def main():
    global img,sw,text
    
    pub = rospy.Publisher('/ocr_pose',ocr_posemsg,queue_size=10)   #发布数据
    rospy.init_node('ocr_pose',anonymous=True)
    rospy.Subscriber('/usb_cam/image_raw',Image,callback)    #接受图像转为cv图像(/usb/cam/image_raw)
    rospy.Subscriber('/ocr_switch',Int8,ocrcallback)          #监听图像检测颜色提取
    
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        
        if sw != 2:
            
            if img != []:
                
                if bool(int(sw)):
                    
                    ocr_pose = ocr_posemsg()
                    
                    capture_pose_x, capture_pose_y, text = text_re.pose_text(img)
                    
                    if type(capture_pose_x) != list and type(capture_pose_y) != list:
                        
                        ocr_pose.x = capture_pose_x
                        ocr_pose.y = capture_pose_y
                        ocr_pose.text = str(text)
                    
                        pub.publish(ocr_pose)
                    else:
                        print("data is an empty !!!")
                    rate.sleep()
                else:
                    
                     ocr_pose = ocr_posemsg()
                     put_pose_x, put_pose_y = cutpig.put_down_transformation(img)
                     
                     if type(put_pose_x) != list and type(put_pose_y) != list:
                           ocr_pose.x = put_pose_x
                           ocr_pose.y = put_pose_y
                    
                           ocr_pose.text = " "
                           pub.publish(ocr_pose)
                     else:
                          print("data is an empty !!!")
                     rate.sleep()
                    
        else:
            print("不侦测")
    rospy.spin()
         


if __name__=='__main__':
    main()    
    
