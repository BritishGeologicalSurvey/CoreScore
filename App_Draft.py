#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:04:36 2019

@author: ziad
"""

import cv2 as cv
import numpy as np
import math
ratio = 3
kernel_size = 45
#@TODO: insert the code used to process the JPEG2000 images
class ProcessImages():
    
    def __init__(self):
        pass
#Main Class with OpenCV Code - there are a couple of globals that need to be inserted as parameters in the GUI
class BoundaryDL():
    #Initialize image read it and scale it to fit screen 
    def __init__(self,fname,ratio,kernel_size,kernels,kernel_pos):
        self.fname = fname
        self.image = cv.imread(fname)
        #Constant scaling at 0.25
        self.image = self.scaleColourImage(0.25,0.25)
        
        #Declare gray image a local variable and convert it 
        self.grayImg = self.grayImage(1,50,kernels,kernel_pos)
        
        self.ratio = ratio
        self.kernelSize = kernel_size
    #Refresh image
    def refreshImage(self,contrast,brightness,kernel, kernel_pos):
        self.image = cv.imread(self.fname)
        self.image = self.scaleColourImage(0.25,0.25)
        self.grayImg = self.grayImage(contrast,brightness,kernel,kernel_pos)
        
    #Main edge detection method 
    def canny(self,thresholdVal):
        #Import gray image
        img = self.grayImg

        low_threshold = thresholdVal
        detected_edges = cv.Canny(img, low_threshold,low_threshold*self.ratio,self.kernelSize)
        
        mask = detected_edges != 0
        dst = img*(mask[:,:].astype(img.dtype))
        
        return dst
    
    #Set brightness and contrast
    def setContrast(self,pic,contrast,brightness):
        pic = cv.addWeighted(pic, contrast, 
                        np.zeros_like(pic),0,brightness -50)
        return pic        
    #Main contouring method
    def contour(self,cannythresh,thresh,img,contrast,brightness,kernel,kernel_pos):
        self.refreshImage(contrast,brightness,kernel, kernel_pos)
        #Process image with canny using threshold
#        img = self.dilate(img)
        img = self.canny(cannythresh)
        
        
        gray_modified = img
        #Gray threshold the canny image
        gray_thresh = cv.adaptiveThreshold(gray_modified,thresh,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        
        #Get contours from given threshold
        image, contours, hierarchy = cv.findContours(gray_thresh,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        image = cv.drawContours(image,contours,-1,(255,255,255),3)
        
        #Bring back image to colour image to display on screen
        img = self.image
        
        
        #Loop through all of the contours - remove ones below a certain value 
        for index,i in enumerate(contours):
            if (cv.contourArea(i)) > 5000:
                try:
                    M = cv.moments(i)
                    cX = int(M["m10"]/M["m00"])
                    cY = int(M["m01"]/M["m00"])
        
                    cv.drawContours(img,contours,index,(0,255,0),2)
                    cv.circle(img,(cX,cY),7,(255,255,255),-1)
                    cv.putText(img,str(round(math.sqrt(cv.contourArea(i)*0.01125),2))+"cm^2", (cX - 20, cY - 20),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                except:
                    print("No suitable contours found")
        return img
    #Scale the colour image (it resets after re-importing the image so needs to be reset sometimes)
    def scaleColourImage(self,scalingFactorX,scalingFactorY):
        x = scalingFactorX
        y = scalingFactorY
        self.color_original = cv.resize(self.image,(0,0),fx=x,fy=y)
        return self.color_original
    #Reprocess gray image after making changes
    def grayImage(self,contrast,brightness,kernel,kernel_pos):
        self.dilated_img = self.dilate(self.image)
        self.GrayImage = self.setContrast(cv.cvtColor(self.dilated_img, cv.COLOR_BGR2GRAY),
                                          contrast,brightness)
        self.GrayImage = cv.filter2D(self.GrayImage, -1, kernel[kernel_pos])
                                          
        return self.GrayImage
    #Method to remove shadows
    def dilate(self,img):
        self.dilated_img = cv.dilate(img,np.ones((7,7),np.uint8))
        self.bg_img = cv.medianBlur(self.dilated_img,21)
        self.diff_img = 255- cv.absdiff(img,self.bg_img)
        self.norm_img = self.diff_img.copy()
        cv.normalize(self.diff_img,self.norm_img,alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        _, thr_img = cv.threshold(self.norm_img, 230, 0, cv.THRESH_TRUNC)
        return thr_img
        
    #To dos below
    def threshold(self):
        pass
    
    def drawContours(self,cannyVal,):
        pass

    def getMidPoint(self):
        pass
    
    def setRatio(self,ratio):
        self.ratio = ratio
    
    def setKernelSize(self,kernelSize):
        self.kernelSize = kernelSize
        
    # def setGFilter(self,filter_,val,img):
            
    #     self.gray_modified = cv.filter2D(self.gray_original,-1,kernels[kernel])

    # def setCFilter(self,filter_,val,img):
    #     self.color_modified = cv.filter2D(self.color_original,-1,kernels[kernel])

## TODO        
class MachineLearn():
    
    def __init__(self):
        pass


#Create GUI..
class GUI():
    
    def __init__(self,trackbars):
        cv.namedWindow("app")
        self.createTrackBars(trackbars)
        #Import image here.. needs to be changed everytime
        self.ImageProcessing("S00128816.Cropped_Top_2.JPEG")
        # self.ImageProcessing("S00128769.Cropped_Top_1.JPEG")

        
        self.destroyWindow()
    #Useless func that is linked to the trackbars  - can run functions direction from trackbar changes
    def dummy(self,val):
        pass
    
    def createTrackBar(self,name,window,startVal,maxVal,onChangeFun):
        #Arguments, trackbarName, Windowname, value(initial value), count (MAX VAL), onchange(event handler)
        cv.createTrackbar(name,window,startVal,maxVal,onChangeFun)
    
    def createTrackBars(self,trackbars,func=None):
        for i in trackbars:
            self.createTrackBar(i[0],i[1],i[2],i[3],self.dummy)
    

    #Initalize the image class
    def ImageProcessing(self,image):
        self.imageClass = BoundaryDL(image,ratio,kernel_size,kernels,0)
    
    def destroyWindow(self):
        while True:
            key = cv.waitKey(100)
            kernel_pos = cv.getTrackbarPos('Filter', 'app')
            print(kernel_pos)
            contrast = cv.getTrackbarPos("Contrast","app")
            brightness = cv.getTrackbarPos("Brightness","app")
            #apply all of the filters and contours 
            
            pic = self.imageClass.contour(cv.getTrackbarPos("Canny","app"),
                                          cv.getTrackbarPos("Threshold","app"),
                                          self.imageClass.image,contrast,
                                          brightness,kernels,kernel_pos)
#                apply brightness and contrast to the colour image
            pic = cv.filter2D(self.imageClass.image, -1, kernels[kernel_pos])
            pic = self.imageClass.setContrast(pic,contrast,brightness)
            
            cv.imshow("app",pic)
            
            
            if key == ord("q"):
                break
#            elif key == ord("t"):
#                pic = self.imageClass.contour(cv.getTrackbarPos("Canny","app"),
#                                          cv.getTrackbarPos("Threshold","app"),
#                                          self.imageClass.image)
#                cv.imshow("app",pic)

        cv.destroyAllWindows()
 
#define convolution kernels

identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
box_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32) / 9.0
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gaussian_kernel1 = cv.getGaussianKernel(3, 0)
gaussian_kernel2 = cv.getGaussianKernel(5, 0)
gaussian_kernel3 = cv.getGaussianKernel(12, 0)

kernels = [identity_kernel, sharpen_kernel,box_kernel,gaussian_kernel1,
           gaussian_kernel2,gaussian_kernel3]   

#All the trackbars are initialized here.. currently global needs to be reworked
Trackbars = [["Contrast","app",1,100],
             ["Brightness",'app',50,100],
             ["Filter",'app',0,len(kernels)-1],
             ["Threshold","app",1,255],
             ["Canny",'app',1,255]]
#Create main GUI 
MainWindow = GUI(Trackbars)
#
#
#
