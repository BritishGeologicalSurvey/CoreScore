#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:31:12 2019

@author: ziad
"""
import os
import logging
import json
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np
from skimage.io import imsave

logging.basicConfig(level=logging.INFO)


# This file does all the pre-processing from the json and into actual masks
# which then can be used directly with the unet implementation in fastai
# We differentiate between rock fragments in Box 1 and 2 (as Corebreakout does)

LABELS = [
    'Void',
    'Rock_Fragment',
    'Rock_Fragment_2',
    'Paper',
    'Core_Plug',
    'Text',
    'Box']


class CoreImageProcessor():
    """
        Class that handles all the preprocessing to make the image
        masks and lower the resolution of the images as defined by a
        constant in the file - some things need to be reworked -
        particularly the JSON file needs to generate the correct
        values for the dictionary as well as the resizing needs to be
        figured out not via constants

        Class takes in the image directory as a string
    """

    def __init__(self, imageDirectory,
                 labels="Core_labels.json",
                 merge_fragment_labels=None,
                 mask_labels=[]):
        """Takes an image directory,
        A set of associated labels for each image filename,
        Optional list of labels to use for masks,
        Optional whether to label Rock_Fragment and Rock_Fragment_2 together
        """

        self.path = os.fspath(Path(imageDirectory))
        self.maskDir = Path(imageDirectory + "/train/")
        with open(labels, 'r') as f:
            self.core_types = json.load(f)
        if mask_labels:
            self.mask_labels = mask_labels
        else:
            self.mask_labels = LABELS

        # Map of greyscale colours to use as masks
        self.masks = {value: index for (index, value) in enumerate(LABELS)}

        if merge_fragment_labels:
            self.masks['Rock_Fragment_2'] = self.masks['Rock_Fragment']

    def getImageNames(self, image):
        """Returns image name from LabelBox compat data"""
        name = None
        if 'External ID' in image:  # LabelBox
            name = image['External ID']
        if 'imagePath' in image:  # LabelTool
            name = image['imagePath']
        return name

    def generateShapes(self, blankImage, shape, values):
        '''Generates the shapes - i.e. box,rock fragments, paper etc and
        returns it into a mask format'''
        return self.genOneMask(blankImage, shape, values)

    def makeBlankCanvas(self, width=6000, height=1200):
        '''makes a blank image to fill in with random data'''
        blank = self.makeBlankImage(width, height)
        return blank

    def getAllImages(self):
        '''returns all of the reviewed labelled images in a list'''
        images = []
        for key in self.core_types:
            if key['Reviews'][0]['score'] >= 1:
                images.append(self.getImageNames(key))
        return images

    def processImage(self, fp):
        '''Main process to generate the masks - all the rest are not really
        neccecary except for the random images'''
        img = self.openImage(os.path.join(self.path,
                                          "train",
                                          self.getImageNames(fp)))
        if not img.any():
            return
        width = img.shape[1]
        height = img.shape[0]
        blank = self.makeBlankImage(width, height)

        mask = self.genMasks(blank, fp)

        filename = self.getImageNames(fp)
        suffix = filename.split('.')[-1]
        filename = filename.replace(suffix, 'png')
        img_path = os.path.join(
            os.getcwd(),
            self.path,
            "train_masks",
            filename)
        logging.info(img_path)
        self.saveMask(img_path, mask)
        return img_path

    def openImage(self, image):
        '''Opens an image and makes it into a numpy array
        '''
        rgbImage = np.empty(0)
        try:
            rgbImage = Image.open(image)
            rgbImage = np.array(rgbImage)
        except FileNotFoundError:
            logging.info(f"no image at {image}")
        return rgbImage

    def makeBlankImage(self, width, height):
        """Generate an image with all 0s
        in the arrays for a given width/height"""
        img = Image.new('L', (width, height), 0)
        return img

    def genShape(self, fp):
        '''Similar to the genMasks function but
        instead returns the shape instead of the mask of
        the image - this is useful for manipulating the geometry
        and or translating the shape afterwards'''
        shapes = {}
        for key in self.masks:
            try:
                for geometry in fp["Label"][key]:
                    shape = self.getXY(*geometry['geometry'])
                    try:
                        shapes[key].append(shape)
                    except BaseException:
                        shapes[key] = [shape]
            except BaseException:
                logging.info("No " + key + " in labelled object")
        return shapes

    def genMasks(self, img, fp):
        """Write masks into the images using the labelled shapes"""
        image = img
        draw = ImageDraw.Draw(image)

        # LabelBox export format
        # "Label": {"Rock_Fragment": [{"geometry": [{"x": 756,
        if 'Label' in fp:
            for key, value in self.masks.items():
                if key in fp["Label"] and key in self.mask_labels:
                    for geometry in fp["Label"][key]:
                        shape = self.getXY(*geometry['geometry'])
                        logging.debug(shape)
                        draw.polygon(shape, outline=value, fill=value)

        # LabelTool export format
        # shapes": [
        #      {
        #    "label": "Rock_Fragment",
        #       "points": [[ 6184, 1223], etc
        elif 'shapes' in fp:
            for shape in fp['shapes']:
                if shape['label'] not in self.mask_labels:
                    continue
                colour = self.masks[shape['label']]
                points = [tuple(p) for p in shape['points']]
                draw.polygon(points, outline=colour, fill=colour)

        return np.array(image)

    def getXY(self, *args):
        '''returns the xy locations for a certain shape/class type'''
        shape = []
        for i in args:
            shape.append((i['x'], i['y']))

        return shape

    def saveMask(self, fname, mask):
        """Save mask to a file"""
        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        imsave(fname, mask)
