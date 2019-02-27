#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 12:41:44 2019

@author: ziad
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#img = cv.imread('Test_Case_2.png',0)
## Initiate FAST object with default values
#fast = cv.FastFeatureDetector_create()
## find and draw the keypoints
#kp = fast.detect(img,None)
#img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
## Print all default params
#print( "Threshold: {}".format(fast.getThreshold()) )
#print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
#print( "neighborhood: {}".format(fast.getType()) )
#print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
#cv.imwrite('fast_true.png',img2)
## Disable nonmaxSuppression
#fast.setNonmaxSuppression(0)
#kp = fast.detect(img,None)
#print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
#img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
#cv.imwrite('Edges_Test_3.png',img3)



im = cv.imread('Test_Case_4.png')
imgray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
thresh = cv.adaptiveThreshold(imgray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

image, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

large_pieces = []
for index,i in enumerate(contours):
    if (cv.contourArea(i)) > 30000:
        print((cv.contourArea(i)*0.15))
        print(index)
        large_pieces.append(index)

for i in large_pieces:
#    if i == 0:
#        pass
#    else:
    cv.drawContours(im,contours,i,(255,255,255),10)
#img = cv.drawContours(im, contours, -1, (255,255,255), 10)
#img = cv.drawContours(im, contours, 45, (255,255,255), 10)
#img = cv.drawContours(im, contours, 529, (255,255,255), 10)


cv.imwrite('Contour_Test_4.png',im)
