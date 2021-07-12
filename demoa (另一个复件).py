#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import re
import json
reload(sys)  
sys.setdefaultencoding('utf8')   

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import tools.infer.utility as utility

from ppocr.utils.utility import initial_logger

logger = initial_logger()
import cv2
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_cls as predict_cls
import copy
import numpy as np
import math
import time
import pypinyin
import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
#from PIL import Image
from tools.infer.utility import draw_ocr
from tools.infer.utility import draw_ocr_box_txt
import cutpig
import ocr_pose
from std_msgs.msg import String


img = []
use_angle_cls = True

class TextSystem(object):
    
    def __init__(self):
        
        self.text_detector = predict_det.TextDetector()
        self.text_recognizer = predict_rec.TextRecognizer()
        self.use_angle_cls = use_angle_cls
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier()

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            print("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)
        print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


#中文转拼音
def pinyin(word):
    word = unicode(word,'utf-8')
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        
        s += ''.join(i)
    return s
#字符匹配
def character_match(text):
    line = ['jiangsu', 'fujian', 'sichuan', 'guangdong', 'yunnan', 'henan', 'anhui',  'hubei', 'hunan', 'shandong']
    
    word = pinyin(str(text))
    print(word)
    for i in range(9):
       text = re.search(line[i],str(word))
       
       if text != None:
          return text.group()
       else:
            
            text = re.search(line[i+1],str(word))
       
#去除字母/数字/标点
def preprocess_English(text):
    text = ''.join(text)
    text = re.sub(r'\W','',text)
    text = text.split(' ')
    text = ''.join(text)
    text = re.sub(r'\d','',text)
    text = re.sub(r'\s','',text)
    
    return text

ocrtext = []
def ocr_totext(img):
    a = 'cuowu'
    global ocrtext
    text_sys = TextSystem()
    
    cutimg = cutpig.cut_coordinate(img)
    
    dt_boxes, rec_res = text_sys(cutimg)
    
    
    rec_res = np.array(rec_res)
    if rec_res != []:
        rec_res = rec_res[:,0]
        rec_res = ''.join(rec_res)
        #rec_res = rec_res.decode('unicode_escape','ignore')
        
        print(111111111111111111111)
        pattern = re.compile(r'[\4e00-\u9fa5]')
        rec_res = re.sub(pattern,'',rec_res)
        print(rec_res)
        print(type(rec_res))
        ocrtotext = character_match(rec_res)
        print(44444444444)
        print(ocrtotext)
        if ocrtotext != []:
            return ocrtotext
        else:
            return ocrtotext == a
        
    else:
        print("data is an empty !!!")

#回调函数将ros图像转为cv图像
def textcallback(data):
    global ocrtext
    pub = rospy.Publisher('/text',String,queue_size=10)   #发布数据
    try:
       image = CvBridge().imgmsg_to_cv2(data, "bgr8")
       #cv2.imshow('image', image)
       #cv2.waitKey(10)
       im = image
       
       if im != []:
           img = np.array(im)
           get_text = ocr_totext(img)
           print(55555555555)
           print(get_text)
           if get_text is not None:
               pub.publish(get_text)
           
           else:
                print("没有文字是正确的！！！")
       else:
            print("不检测")
       return im
    except CvBridgeError as e:
       print(e)



def main():
    
    
    rospy.init_node('demoa',anonymous=True)
    pub = rospy.Publisher('/text',String,queue_size=10)   #发布数据
    
    rospy.Subscriber('/usb_cam/image_raw',Image,textcallback)    #接受图像转为cv图像(/usb/cam/image_raw)
    
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        rate.sleep()
    rospy.spin()
            
   
                
        
    
if __name__ == "__main__":
    main()




