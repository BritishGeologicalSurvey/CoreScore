# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:03:08 2021

@author: Alex
"""


import cv2

from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
import os

TEST_DIR = '../Images/difficult_images'
PREDICTIONS_DIR = '../Images/results/difficult_image_predictions'
MASK_DIR = '../Images/results/test_masks'
ROCK_LABEL = 1
COMPUTE_METRICS = False
INFER_ONLY = True


test_size = len(os.listdir(TEST_DIR)) 

if(INFER_ONLY):
    cols = ['Raw Image', 'Prediction']
    f, axarr = plt.subplots(test_size,2, figsize = (7,12))
    for ax, col in zip(axarr[0], cols):
        ax.set_title(col)
    f.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    
else:    
    cols = ['Raw Image', 'Actual (mask)', 'Prediction']
    f, axarr = plt.subplots(test_size,3, figsize = (8,12))
    for ax, col in zip(axarr[0], cols):
        ax.set_title(col)
    f.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

if(COMPUTE_METRICS):
    cols = ['Raw Image', 'Actual (mask)', 'Rock False Positives', 'Rock False Negatives']
    f1, axarr1 = plt.subplots(test_size,4, figsize = (11,12))
    for ax, col in zip(axarr1[0], cols):
        ax.set_title(col)    
    f1.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

i=0
accuracy_list = []
rock_accuracy_list = []
false_positive_list = []
false_negative_list = []

for f in os.listdir(TEST_DIR):
    
    f_name = str.split(f,'.')[0]
    
    pred = image.imread(os.path.join(PREDICTIONS_DIR, f_name + '.bmp'))
    original = image.imread(os.path.join(TEST_DIR, f_name + '.jpg'))
    #upscale prediction to match original
    pred = cv2.resize(pred, dsize = (np.shape(original)[1], np.shape(original)[0]), interpolation = cv2.INTER_NEAREST)
    
    if(INFER_ONLY == False):
        mask = image.imread(os.path.join(MASK_DIR, f_name + '.png'))
        
        #resize prediction to the mask - there can sometimes be a minor disparity between the mask and original
        pred = cv2.resize(pred, dsize = (np.shape(mask)[1], np.shape(mask)[0]), interpolation = cv2.INTER_NEAREST)
        #convert mask to integers
        mask=mask*255
        mask=mask.astype(int)
        
    
    

    

    
    if(INFER_ONLY):
        axarr[i,0].imshow(original)
        axarr[i,0].axis('off')
        axarr[i,1].imshow(pred)
        axarr[i,1].axis('off')
        
    
    else:
        axarr[i,0].imshow(original)
        axarr[i,0].axis('off')
        axarr[i,1].imshow(mask)
        axarr[i,1].axis('off')
        axarr[i,2].imshow(pred)
        axarr[i,2].axis('off')
    
    
    if(COMPUTE_METRICS):
        mask_rock = mask
        pred_rock = pred
        mask_rock[mask_rock>1]=0
        pred_rock[pred_rock>1]=0
        pred_not_rock = pred==0
        
        true_pos_filter = pred_rock+mask_rock
        false_pos_filter = pred_rock - mask_rock
        false_neg_filter = mask_rock - pred_rock
        
        
        rock_true_positives = true_pos_filter==2
        rock_false_positives = false_pos_filter==1
        rock_false_negatives = false_neg_filter==1
        
        axarr1[i,0].imshow(original)
        axarr1[i,0].axis('off')
        axarr1[i,1].imshow(mask)
        axarr1[i,1].axis('off')
        axarr1[i,2].imshow(rock_false_positives)
        axarr1[i,2].axis('off')
        axarr1[i,3].imshow(rock_false_negatives)
        axarr1[i,3].axis('off')
        
        rock_accuracy = np.sum(rock_true_positives)/np.sum(mask_rock)
        false_pos_rate = np.sum(rock_false_positives/np.sum(pred_rock))
        false_neg_rate = np.sum(rock_false_negatives/np.sum(pred_not_rock))
        
        
        total_accuracy = np.sum(pred==mask) / (np.shape(mask)[1] * np.shape(mask)[0])
        accuracy_list.append(total_accuracy)
        rock_accuracy_list.append(rock_accuracy)
        false_positive_list.append(false_pos_rate)
        false_negative_list.append(false_neg_rate)
    
    i = i+1
    
    
if(COMPUTE_METRICS):
    accuracy = sum(accuracy_list) / len(accuracy_list)
    rock_true_positive = sum(rock_accuracy_list)/len(rock_accuracy_list)
    rock_false_positive = sum(false_positive_list)/len(false_positive_list)
    rock_false_negative = sum(false_negative_list)/len(false_negative_list)


