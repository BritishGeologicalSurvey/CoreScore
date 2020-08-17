#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:20:34 2019

@author: ziad
"""
# Code here is for downloading the premade masks that labelbox provides
# Takes keys form Json provided by labelbox and does various things
# Everything here is a helper function and is not really needed unless
# Labelbox is being used from scratch again


import glob
import re
from PIL import Image
import os
#cur_dir = os.getcwd()

#file_list = os.listdir(cur_dir)

#file_list2 = os.listdir('masks/')
# print(file_list)
# for i in file_list:
#    print(i)

import urllib.request
import json


with open("Core_labels4.json", 'r') as f:
    core_types = json.load(f)


def mkdir(dirname, file):
    '''Make directories for images to be saved'''
    if os.path.isfile(dirname):
        os.rename(file, dirname + "/" + file)
    else:
        try:
            os.mkdir(dirname)
            os.rename(file, dirname + "/" + file)
        except BaseException:
            os.rename(file, dirname + "/" + file)

# print(len("S00128933.Cropped_Top_1.JPEG"))


def downloaderMasks(key):
    ''' Does what it says on the tin - takes in keys from json
    as input and downloads the masks from labelbox note
    that if masks have not been reviewed - they are not downloaded'''
       try:
            if key['Reviews'][0]['score'] >= 1:

                url = (key['Masks']['Rock_Fragment'])
                fname = key['External ID']
                fname = (fname[:-4])
                urllib.request.urlretrieve(
    url, 'data/train_masks/' + fname + "png")
        except BaseException:
            print("No mask available")


def downloaderImages(key):
    ''' Does what it says on the tin - takes in keys from json
    as input and downloads the images from labelbox note
    that if images have not been reviewed - they are not downloaded
    '''
       try:
    #            print(key['Reviews'][0]['score'])
            if key['Reviews'][0]['score'] >= 1:

                url = (key['Masks']['Rock_Fragment'])
                fname = key['External ID']
                url = (key['Labeled Data'])
                fname = (fname[:-4])
                urllib.request.urlretrieve(url, 'data/train/'+fname+"JPEG")
        except BaseException:
            print("No mask available")
# For classification problems - ended up not working so commented out -
# Could be useful for some other problems


# for key in core_types:
#    downloaderImages(key)
#    downloaderMasks(key)
#    for fname in file_list:
#        if fname == key['Labeled Data'][-28:]:
#
#            print(fname)
#            if key["Label"]['pick_most_suitable_classification_for_rock_condition'] == 'excellent':
#                mkdir("0",fname)
#            elif key["Label"]['pick_most_suitable_classification_for_rock_condition'] == 'moderate':
#                mkdir("1",fname)
#            elif key["Label"]['pick_most_suitable_classification_for_rock_condition'] == 'poor':
#                mkdir("2",fname)
#            elif key["Label"]['pick_most_suitable_classification_for_rock_condition'] == 'very_poor':
#                mkdir("3",fname)

def convert_ToGIF(file_list):
    '''convert the files to gifs - one of the examples
    was using gifs so this was just to test'''
    for i in file_list2:
        print(i)
        im = Image.open("masks/" +i)
        im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
    #    transparency = im.info['transparency']
        im .save("masks/" +i[:-3] +'.gif')


def convert_ToPNG(file_list):
    '''Files need to be converted into pngs
    for fastai to work - at least with the current
    code base'''
    for i in file_list:
        print(i[-4:].upper() is "JPEG")
        if i[-4:] == "JPEG":
            print("Processing file:..." +i)
            im = Image.open("data/train/" +i)
        #    transparency = im.info['transparency']
            im .save("data/train/" +i[:-4] +'png')


def download_files(core_types):
    '''Download the images and or masks'''
    for key in core_types:
    #        print(key)
        downloaderImages(key)
#        downloaderMasks(key)


# download_files(core_types)
convert_ToPNG(os.listdir("data/train/"))
