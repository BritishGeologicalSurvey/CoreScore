#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:31:12 2019

@author: ziad
"""
import os
import glob
import re
import logging
import json

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path
from scipy.ndimage.interpolation import rotate

# Sklearn resiez image libraries
from skimage import io
from skimage.transform import resize
from skimage import img_as_bool
from skimage.io import imsave

import random
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)

with open("Core_labels.json", 'r') as f:
    core_types = json.load(f)

# This file does all the pre-processing from the json and into actual masks
# which then can be used directly with the unet implementation in fastai
# a lot of the code is not needed - some is used to create synthetic datasets
# Should probably be re-written in the long run- a lot of the functions do
# Similar things
# Import image with PIL and then get the bands - i.e. shape + rgb
#IMAGE = "S00128906.Cropped_Top_2.JPEG"
CMAP = "tab20"
RESIZE_SIZE = (89, 629)


class coreImageProcessor():
    '''
        Class that handles all the preprocessing to make the image
        masks and lower the resolution of the images as defined by a
        constant in the file - some things need to be reworked -
        particularly the JSON file needs to generate the correct
        values for the dictionary as well as the resizing needs to be
        figured out not via constants

        Class takes in the image directory as a string
    '''

    def __init__(self, imageDirectory):
        '''Takes an image directory - and prepares the dictionary for
        creating the masks, ideally the dictionary will be stored in a json
        somewhere but at the moment there isnt enough classes to warrant a
        larger file'''

        #self.path = os.fspath(Path(imageDirectory))
        self.path = imageDirectory
        self.maskDir = Path(imageDirectory + "/train/")
        self.dicto_ = OrderedDict({"Box": [0, 0], "Rock_Fragment": [1, 1], "Paper": [
                                  2, 2], "Core_Plug": [3, 3], "Text": [4, 4]})

    def getImageNames(self, key):
        '''returns image name'''
        return key['External ID']

    def generateShapes(self, blankImage, shape, values):
        '''Generates the shapes - i.e. box,rock fragments, paper etc and
        returns it into a mask format'''
        return self.genOneMask(blankImage, shape, values)

    def rotateMask(self, blankImage, mask, originalImage, rotation):
        '''Function used to rotate everything - used to increase
        randomness in synthetic dataset - works but not so well'''
        blankImage = rotate(blankImage, rotation)
        mask = rotate(mask, rotation)
        originalImage = rotate(originalImage, rotation)
        return blankImage, mask, originalImage

    def genRandomColour(self):
        """Returns a random colour in the RGB Spectrum"""
        colour = list(np.random.choice(range(256), size=3))
        return colour

    def getNumPiecesInBox(self, shape_type, dicto):
        '''returns the number of items of a certain class i.e. rock paper plug
        etc that lie within a certain labelled image'''
        return len(dicto[shape_type])

    def getRandomPiece(self, shape_type, shapes, blank):
        '''returns a random piece from the dictionary'''
        randomPiece = (
            np.random.randint(
                0, self.getNumPiecesInBox(
                    shape_type, shapes)))
        Onemask = self.genOneMask(
            blank,
            shapes[shape_type][randomPiece],
            (self.dicto_[shape_type][0],
             self.dicto_[shape_type][1]))
        return Onemask, shapes[shape_type][randomPiece]

    def makeBlankCanvas(self, width=6000, height=1200):
        '''makes a blank image to fill in with random data'''
        blank = self.makeBlankImage(width, height)

        return blank

    def getAllImages(self):
        '''returns all of the reviewed labelled images in a list'''
        images = []
        for key in core_types:
            if key['Reviews'][0]['score'] >= 1:
                images.append(self.getImageNames(key))
        return images

    def addNoise(self, image):
        '''add chaos to a given image - randomness values are hardcoded'''
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.9
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

      # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    def makeNewImage(self, image):
        '''makes a new image with noise - based on the dimensions of another
        input image - needs to be done so that the images come out same size
        with similar aspect ratios - basically we can only generate random
        images if all of the pieces come from the same size image'''
        newImage = np.zeros(image.shape)
        newImage = np.random.random(image.shape)
        newImage[:, :] = self.genRandomColour()
        newImage = self.addNoise(newImage)

        return newImage

    def getRandomImage(self):
        '''returns a random image from all the labelled images'''
        images = self.getAllImages()
        return np.random.choice(images)

    def setupImage(self, makeBlank=False):
        '''setup an image for creating a random one'''
        fp = self.getRandomImage()
        img = self.openImage(self.path + "/train/" + fp)
        width = img.shape[1]
        height = img.shape[0]
        blank = self.makeBlankImage(width, height)
        if makeBlank:
            blank = self.makeBlankImage(width, height)
            newImg = self.makeNewImage(img)
            return fp, img, newImg, width, height, blank
        return fp, img, width, height, blank

    def getKey(self, name):
        '''returns the key given image name'''
        for key in core_types:
            if self.getImageNames(key) == name:
                return key

    def generateLotsofRandomImages(self):
        '''generates 10 random images'''
        for i in range(10):
            self.makeRandomImages()

    def makeRandomImages(self):
        '''makes a random image'''
        fp, img, newImg, width, height, blank = self.setupImage(makeBlank=True)
        key = self.getKey(fp)
        originalH = height
        originalW = width
        self.mask = self.persistentMask(originalW, originalH)
        for i in range(10):
            newImg = self.extractRGBVals(key, img, blank, newImg)

            fp, img, width, height, blank = self.setupImage()
            while height != originalH:
                fp, img, width, height, blank = self.setupImage()
            key = self.getKey(fp)
        randomString = ''.join(random.sample(fp, len(fp)))
        self.saveMask(self.path + "/train/" + randomString + ".", newImg)
        self.saveMask(
            self.path +
            "/train_masks/" +
            randomString +
            ".",
            self.mask)
        plt.imshow(newImg / 255)

    def getRandomPieces(self, fps):
        pass

    def extractRGBVals(self, fp, img, blank, newImg, resize=False):

        y, x, z = (img.shape)
        print(x, y)
        shape = self.genShape(self.dicto_, fp)

        oneMask, randomPiece = self.getRandomPiece(
            'Rock_Fragment', shape, blank)
        if resize is True:
            shape = self.genShape(self.dicto_, fp)

            oneMask = self.getRandomPiece('Rock_Fragment', shape, blank)
            oneMask = np.ma.resize(oneMask, (y, x))

        # Convert the image to a 0-255 scale.
            rescaled_image = 255 * img
        # Convert to integer data type pixels.
            img = rescaled_image.astype(np.uint8)
            #newImg = cv.resize(newImg,(x,y),interpolation=cv.INTER_CUBIC)
#            oneMask = resize(oneMask,(img.shape[0],img.shape[1]))
        mask = oneMask == 1


#        newImage,mask,img = self.rotateMask(newImg,mask,img,np.random.choice([0,180]))

        newImg[mask] = img[mask]
        print(randomPiece)
        randomNumber = np.random.randint(0, 6000)
        self.mask = self.genOneMask(self.mask, randomPiece, (1, 2))
#        print(self.getMinShape(randomPiece))
#        self.mask = self.shift(self.mask,randomNumber)
#        newImg = self.shift(newImage,randomNumber)
        # Need to do this to make the plotting work

#        plt.savefig("Mask_Test.pdf",figsize=(40,20))
#        print(mask)
        return newImg

    def persistentMask(self, width, height):
        self.mask = self.makeBlankImage(width, height)
        return self.mask

    def transformMask(self, mask, number):
        '''move the mask - uses shift'''

        return self.shift(mask, number)

    def shift(self, arr, num):
        '''shift a mask by a given number'''
        arr = np.roll(arr, num)
        if num < 0:
            np.put(arr, range(len(arr) + num, len(arr)), np.nan)
        elif num > 0:
            np.put(arr, range(num), np.nan)
        return arr

    def processImage(self, fp):
        '''Main process to generate the masks - all the rest are not really
        neccecary except for the random images'''
        img = self.openImage(self.path + "/train/" +
                             str(self.getImageNames(fp)[:-4]) + "png")
        if not img.any():
            return
        width = img.shape[1]
        height = img.shape[0]
        blank = self.makeBlankImage(width, height)
        rgbImageResized = resize(img, RESIZE_SIZE)
#        self.saveMask(rgbImageResized,self.path+"/train/"+str(fp))
#        print(type(fp))
        mask = self.genMasks(self.dicto_, blank, fp)
        resizedMask = resize(mask, RESIZE_SIZE)
#        self.saveMask(self.path+"/train_masks/"+fp[:-4],resizedMask)
        shape = self.genShape(self.dicto_, fp)
#        self.genConvexPoly(shape,img)
#        print(self.translateShape(shape,self.getMinShape(shape)))
        filename = self.getImageNames(fp)
        filename = filename.replace('JPEG', '')
        img_path = os.path.join(os.getcwd(), self.path, "train_masks", filename)
        logging.info(img_path)
        self.saveMask(img_path, mask)

    def genConvexPoly(self, shape, img):
        '''Not actually used'''
        coords = self.getMaxShape(shape)
        blank = np.zeros((coords[0], coords[1]))
        shape = np.array(shape)
        cv.fillConvexPoly(blank, shape, 1)

        blank = blank.astype(np.bool)
        img = resize(img, (coords[0], coords[1]))
        out = np.zeros_like(img)
        out[blank] = img[blank]
        cv.imwrite('output.png', out)

    def openImage(self, image):
        '''Opens an image and makes it into a numpy array
        - also assumes that the image being opened is a jpeg
        (since thats what we're using here)
        '''
        rgbImage = np.empty(0)
        try:
            rgbImage = Image.open(image)
            rgbImage = np.array(rgbImage)
        except FileNotFoundError:
            logging.info(f"no image at {image}")
        return rgbImage

    def PolyArea(self, x, y):
        '''
        Calculate area of polygon using shoelace formula in numpy
        '''
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                            np.dot(y, np.roll(x, 1)))

    def makeBlankImage(self, width, height):
        ''' Generate an image with all 0s in the arrays for a given width/height'''
        img = Image.new('L', (width, height), 0)
        return img

    def genShape(self, dicto, fp):
        '''Similar to the genMasks function but
        instead returns the shape instead of the mask of
        the image - this is useful for manipulating the geometry
        and or translating the shape afterwards'''
        shapes = {}
        for key, values in dicto.items():
            try:
                for geometry in fp["Label"][key]:
                    shape = self.getXY(*geometry['geometry'])
                    try:
                        shapes[key].append(shape)
                    except BaseException:
                        shapes[key] = [shape]
            except BaseException:
                print("No " + key + " in labelled object")
        return shapes

    def getMinShape(self, shape):
        '''Get minimum value in shape -
        essentially a helper function for the translation'''
        minX = None
        minY = None
        for value in shape:
            if minX is None:
                minX = value[0]
                minY = value[1]
            elif minX > value[0]:
                minX = value[0]
                if minY > value[1]:
                    minY = value[1]
            elif minY > value[1]:
                minY = value[1]
            else:
                pass
        return (minX, minY)

    def getMaxShape(self, shape):
        '''returns the maximum values in a shape - so the farthest locations
        for a given shape'''
        maxX = None
        maxY = None
        for value in shape:
            if maxX is None:
                maxX = value[0]
                maxY = value[1]
            elif maxX < value[0]:
                maxX = value[0]
                if maxY < value[1]:
                    maxY = value[1]
            elif maxY < value[1]:
                maxY = value[1]
            else:
                pass
        return (maxX, maxY)

    def translateShape(self, shape):
        '''Translate the shape in order to have all
        the shapes start at 0,0 origin, this will ensure that while
        generating synthetic datasets that they are relatively well
        aligned'''
        minVal = self.getMinShape(shape)
        tfd_Shape = []
        for value in shape:
            tfd_Shape.append(((value[0] - minVal[0]), (value[1] - minVal[1])))
        return tfd_Shape

    def genOneMask(self, img, shape, values):
        '''generates one mask instead of many - used for creating random images'''
        try:
            ImageDraw.Draw(img).polygon(
                shape, outline=values[0], fill=values[1])
        except BaseException:
            print("Drawing mask failed, trying from array")
            img = Image.fromarray(img)
            ImageDraw.Draw(img).polygon(
                shape, outline=values[0], fill=values[1])
        return np.array(img)

    def genMasks(self, dicto, img, fp):
        """Write masks into the images using the labelled shapes"""
        image = img
        for key, values in dicto.items():
            if key in fp["Label"]:
                for geometry in fp["Label"][key]:
                    shape = self.getXY(*geometry['geometry'])
                    logging.debug(shape)
                    ImageDraw.Draw(image).polygon(
                        shape, outline=values[0], fill=values[1])

        return np.array(image)

    def getXY(self, *args):
        '''returns the xy locations for a certain shape/class type'''
        shape = []
        for i in args:
            shape.append((i['x'], i['y']))

        return shape

    def saveMask(self, fname, mask, ext='png'):
        '''Uses scipy to save the img as a png - method going to be
        removed from the next scipy version - need to find workaround'''
        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        imsave(fname + ext, mask)


coreProcessor = coreImageProcessor(
    "Images")
# print(coreProcessor.generateLotsofRandomImages())
# print(core_types.keys())
for key in core_types:
    #    print(type(key))
    if key['Reviews'][0]['score'] >= 1:
        coreProcessor.processImage(key)

# print(key)
# coreProcessor.extractRGBVals(key)
# break
#        pass
#        break
# Tests for showing the image masks in matplotlib

# Also needed for the mask
#geom = genMasks()
# for key,val in dicto_.items():
#    print(key,val[0],val[1])
#    genMasks(key,val[0],val[1])


# print(geom)
#x = []
#y = []
# for i in geom:
# print(i)
#    x.append(i[0])
#    y.append(i[1])
#
# print(np.array(x))
# print(np.array(y))
# genMasks()


# You need this for the mask
#mask = np.array(img)


# print(mask.dtype)

# print(mask)
#plt.imshow(mask, cmap="gray")
# plt.show()
# print(PolyArea(x,y)*0.016)


# Do plotting and magic
#resizedMask = (resize(mask, (89, 629)))

#rescaledMask = 255 * resizedMask
# Convert to integer data type pixels.
#final_Mask = rescaledMask.astype(np.uint8)


# for i in final_Mask:
#    for x in i:
#        print(x)

#rgbImage = Image.fromarray(rgbImage)

#rgbImageResized = resize(rgbImage,(89,629))
# rgbImageResized.save("test1.jpeg")
#scipy.misc.imsave('outfile.jpg', rgbImageResized)
# final_Mask.save("test1.png")
#scipy.misc.imsave('outfile_mask.png', final_Mask)

#fig, (ax0, ax1) = plt.subplots(1, 2,figsize=(72,48))
#
# ax0.imshow(rgbImage)
#ax0.imshow(mask, cmap=CMAP, alpha = 0.6)
#ax0.set_title('Boolean array')
#
#
# ax1.imshow(rgbImageResized)
#ax1.imshow(final_Mask, cmap=CMAP,alpha = 0.6)
# ax1.set_title('Resized')
#
#
# plt.savefig("Masks_Only.pdf")
# plt.show(fig)
#
##
#
