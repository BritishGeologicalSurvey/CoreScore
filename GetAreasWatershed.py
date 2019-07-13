#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:42:44 2019

@author: ziad
"""
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import imutils

import cv2 as cv

import numpy as np

from pathlib import Path

data_folder = Path("PathToImages")


#IMAGE = "S00128930.Cropped_Top_1.png"
def getBasins(imageInput):
    '''Do watershed segmentation using the masks from the machine learning
    algorithm - takes in everything and segments it including pore plugs
    etc.. needs to be modified to take only rocks mask - commented code
    is for smoothing and trying to deal with the artifacts from the ML 
    model'''
    image = cv.imread(str(imageInput)[:-4]+"_resized"+str(imageInput)[-4:],0)
    mask = cv.imread(str(imageInput),0)
#    newMask = mask == 1
    kernel = np.ones((3,3),np.uint8)
    #opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    #kernel = np.ones((1,1),np.uint8)
    
    #closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    #result = cv.resize(result,((1000),400))
    #kernel = np.ones((15,15),np.float32)/225
        
    #smoothed = cv.filter2D(mask,-1,kernel)
    smoothed = mask
#    result = cv.bitwise_and(image, image, mask=smoothed)
    
    
    
    
    
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(smoothed)
    localMax = peak_local_max(D, indices=False, min_distance=1,
    	labels=smoothed)
    
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=smoothed)
    #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    image = cv.imread(str(imageInput)[:-4]+"_resized"+str(imageInput)[-4:])
    #image = result
    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        if label == 0:
            pass
        else:
    #        mask = np.zeros(result.shape, dtype='uint8')
    #        mask[labels == label] = 255
            cnts = cv.findContours(smoothed.copy(), cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE)
    
    
            cnts = imutils.grab_contours(cnts)
    return cnts,image
def grabContours(cnts,image):
    '''Returns rock areas from contours given - 
    pass onto csv writer to output the results'''
    counter = 0
    Rock_areas = []
    allConts = []
    for index,contour in enumerate(cnts):
        if cv.contourArea(contour) > 100:
            M = cv.moments(contour)
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
    
            cv.drawContours(image,cnts,index,(0,255,0),2)
            cv.circle(image,(cX,cY),7,(255,255,255),-1)
            cv.putText(image,"# "+str(counter)+ " "+str(round(cv.contourArea(contour),2))+"px", (cX - 20, cY - 20),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
            counter +=1
            print("Number of rocks..." +str(counter))
            
            Rock_areas.append(cv.contourArea(contour))
            allConts.append(contour)
    print(len(Rock_areas))

    return Rock_areas,image,allConts
#                cv.drawContours(image,cnts,index,(0,255,0),2)
#for i in Rock_areas:
#    print(i)
#
#areas,image, geometry = grabContours(*getBasins(IMAGE))
import csv
def getValues(image):
    '''Use the previous functions using only an image path as a string,
    returns a contoured image incase it needs to be shown using open cv-
    save paths are hard coded -needs to be fixed'''
    areas,img, geometry = grabContours(*getBasins(image))
    
    with open(str(image)[:-4]+'.csv', 'w') as csvfile:
        CSVwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for index,rock in enumerate(areas):
            
            CSVwriter.writerow([index , rock ])
    cv.imwrite("PathToImages/Contoured/"+str(image.stem)+"_Countoured"+".png", img) 
    return img
### Run through the files and create contoured images and save the areas as a csv
for child in data_folder.glob('*.png'): 
    print(child)
    if "resized" in str(child):
        pass
    else:
        image = getValues(child)

#image = getValues(IMAGE)
#
#
#
