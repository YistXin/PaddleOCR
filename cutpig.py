## -*- coding: utf-8 -*-
import sys


import cv2
import numpy as np
import time
from math import sqrt
ball_color = 'yellow'

color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              'yellow':{'Lower': np.array([0, 70, 70]), 'Upper': np.array([100, 255, 255])},

              }


#cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

#实际图像像素与实际距离转换
def pixel_to_distance(width):
    acttual_distance = 0.09500 / width


    return acttual_distance


#去畸变函数
def correct_img(img):
    camera_matrix = [[371.450355, 0.000000, 320.451676],
                     [0.000000, 494.574368, 234.028774],
                     [0.000000, 0.000000, 1.000000]]          #摄像头内部矩阵
    mtx = np.float32(camera_matrix)

    h, w = img.shape[:2]
    dist          = [-0.347383, 0.081498, 0.004733, -0.001698, 0.000000]     #畸变矩阵
    dist = np.float32(dist)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 自由比例参数

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst


#转换图像实际坐标x,y
def coordinate_transformation(frame):
    actural_central_x = []
    actural_central_y = []
    if frame is not None:
        # frame = cv2.flip(frame,0)
        # 进行图像去畸变，中心点重置，设为图像的中心点
        img1 = frame
        img1 = correct_img(img1)                       #图像去畸变
        #img1 = cv2.flip(img1, 0)                       #垂直反转，将图像转正

        gs_frame = cv2.GaussianBlur(img1, (5, 5), 0)  # 高斯模糊
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
        erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
        inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        if cnts != []:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print(box)
            # 坐标原点左上角，以离原点坐标最近为第一个点，顺时针方向排序
            # 中心坐标转换
            box1 = box[1]
            box2 = box[2]
            box3 = box[3]
            box4 = box[0]

            a_point_x = box1[0] - 320
            b_point_x = box2[0] - 320
            c_point_x = box3[0] - 320
            d_point_x = box4[0] - 320
            a_point_y = box1[1] - 240
            b_point_y = box2[1] - 240
            c_point_y = box3[1] - 240
            d_point_y = box4[1] - 240
            if c_point_x > a_point_x:
                central_point_x = (a_point_x + c_point_x) * 0.5
                central_point_y = (a_point_y + c_point_y) * 0.5
                
                width = sqrt((b_point_x - a_point_x) ** 2 + (b_point_y - a_point_y) ** 2)      #像素距离
                
                l = pixel_to_distance(width)               #像素与图像比例
                actural_central_x = -central_point_x * l  
                actural_central_y = central_point_y * l + 0.05
                #画框
                #cv2.drawContours(img1, [np.int0(box)], -1, (0, 255, 255), 2)
        else:
            print("image is an empty !!!")        
            
                
    #cv2.imshow('correct', img1)
    #cv2.waitKey(10)
    return actural_central_x, actural_central_y



# 提取图像矫正
def cr_img(frame):
    result_img = []
    if frame is not None:
        # 进行原图矫正
        img = frame
        # img = cv2.flip(img, 1)
        # img = cv2.flip(img, 0)  # 垂直反转，将图像转正
        gs_frame = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯模糊
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
        erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
        inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if cnts != []:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            angle = rect[2]   
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            result_img = cv2.warpAffine(img, M, (cols, rows))


    return result_img




#提取图像中黄色区域并裁剪
def cutimg(frame):
    
    img = []
    if frame is not None:
        # 进行原图分离黄色区域并剪裁，进行文字识别
        img = frame
        img1 = frame.copy()
        gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
        erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
        inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if cnts != []:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 原图剪裁
            box1 = box[1]
            box2 = box[2]
            box3 = box[3]
            box4 = box[0]
            a_point_x1 = box1[0]
            b_point_x1 = box2[0]
            c_point_x1 = box3[0]
            d_point_x1 = box4[0]
            a_point_y1 = box1[1]
            b_point_y1 = box2[1]
            c_point_y1 = box3[1]
            d_point_y1 = box4[1]

            central_point_x1 = (a_point_x1 + c_point_x1) * 0.5
            central_point_y1 = (a_point_y1 + c_point_y1) * 0.5

            x = int((c_point_x1 - a_point_x1))
            y = int((c_point_y1 - a_point_y1))

            x1 = central_point_x1 - a_point_x1
            y1 = central_point_y1 - b_point_y1
            S = x * y
            if S != 0 :
                if central_point_y1 - y1 < 0 or central_point_x1 - x1 < 0:
                    img = img[0:int(central_point_y1 + y1),
                          0:int(central_point_x1 + x1)]
                else:
                    img = img[int(central_point_y1 - y1):int(central_point_y1 + y1),
                              int(central_point_x1 - x1):int(central_point_x1 + x1)]
                # 画框
                #cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)
                #cv2.imshow('cut', img)
                #cv2.waitKey(10)
                #img = img_resize_to_target_white(img)


    #cv2.imshow('yuantu', frame)
    #cv2.waitKey(10)
    return  img



#转换释放蓝色图像实际坐标x,y
def put_down_transformation(frame):
    actural_central_put_x = []
    actural_central_put_y = []
    
    if frame is not None:
        # frame = cv2.flip(frame,0)
        # 进行图像去畸变，中心点重置，设为图像的中心点
        
        img = correct_img(frame)                       #图像去畸变
        #img = cv2.flip(img, 0)                       #垂直反转，将图像转正

        gs_frame = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯模糊
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
        erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
        inRange_hsv = cv2.inRange(erode_hsv, color_dist['blue']['Lower'], color_dist['blue']['Upper'])
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if cnts != []:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print(box)
            # 坐标原点左上角，以离原点坐标最近为第一个点，顺时针方向排序
            # 中心坐标转换
            box1 = box[1]
            box2 = box[2]
            box3 = box[3]
            box4 = box[0]

            a_point_x = box1[0] - 320
            b_point_x = box2[0] - 320
            c_point_x = box3[0] - 320
            d_point_x = box4[0] - 320
            a_point_y = box1[1] - 240
            b_point_y = box2[1] - 240
            c_point_y = box3[1] - 240
            d_point_y = box4[1] - 240
            if c_point_x > a_point_x:
                central_point_x = (a_point_x + c_point_x) * 0.5
                central_point_y = (a_point_y + c_point_y) * 0.5
                
                width = sqrt((b_point_x - a_point_x) ** 2 + (b_point_y - a_point_y) ** 2)      #像素距离
                
                l = pixel_to_distance(width)               #像素与图像比例
                actural_central_put_x = -central_point_x * l  
                actural_central_put_y = central_point_y * l + 0.05
                #画框
                #cv2.drawContours(img, [np.int0(box)], -1, (0, 255, 255), 2)
        else:
            print("image is an empty !!!")     
                
    #cv2.imshow('correct', img)
    #cv2.waitKey(10)
        
    return actural_central_put_x, actural_central_put_y

#获取实际坐标及剪切图像
def cut_coordinate(frame):
    
    if frame != []:
        img = frame.copy()
        actural_central_x, actural_central_y = coordinate_transformation(frame)
        #img = cr_img(img)
        img = cutimg(img)
        #img = cv2.flip(img,0)
        
    return  actural_central_x, actural_central_y, img
