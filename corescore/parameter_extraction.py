from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage import img_as_ubyte
import numpy as np
import pandas as pd
import math
import os
import logging
import sys


CORE_PARAMETERS = ["relative_regions_area", "total_regions_perimeter",
                   "avg_region_perimeter", "number_of_regions",
                   "total_regions_area", "perimeter_complexity"]



class Image():
    def __init__(self, filename, min_region, max_region):
        self.filename = filename
        self.min_region = min_region
        self.max_region = max_region
        self.basename = os.path.basename(filename)
        self.image_mask = self.color_mask(filename)
        self.extract_regions()
        
        
    def get_sample_name(self):
        """get the sample name from the filename"""
        sample_name = self.basename.split('_')[0]
        return sample_name

    
    def color_mask(self, img_file):
        """convert image to binary with rock or not"""
        

        
        try:
            self.image = imread(img_file)
        except FileNotFoundError as err:
            msg = "Please supply a filename that exists"
            raise FileNotFoundError(msg) from err
            
        binary_img = self.image
        binary_img[binary_img>1]=0

        return binary_img.astype(int)

    def image_bw(self):
        """Returns a black/white rendering of the image mask"""
        mask = self.image_mask
        mask[mask > 0] = 255
        return img_as_ubyte(mask)

    def extract_regions(self):
        """returns input-image sized array with distinct regions labelled
        ordinally"""

        self.labels = label((self.image_mask), background = 0)
        props = regionprops(self.labels)
        # filter out region within a specified size range
        self.regions = list(filter(lambda region: region.area > self.min_region and
                                 region.area < self.max_region, props))
    


    def total_regions_area(self, min_area=None):
        areas = [region.area for region in self.regions]
        if min_area:
            list(filter(lambda x: x > min_area, areas))
        return sum(areas)

    def relative_regions_area(self, min_area=None):
        areas = [region.area for region in self.regions]
        if min_area:
            region_area = self.absolute_area()

        region_area = sum(areas)
        x, y = self.image_mask.shape
        total_area = x * y
        return region_area / total_area

    def total_regions_perimeter(self):
        return sum([region.perimeter for region in self.regions])

    def avg_region_perimeter(self):
        try:
            avg_perimeter = self.total_regions_perimeter() / len(self.regions)
        except ZeroDivisionError:
            avg_perimeter = None
        return avg_perimeter

    def number_of_regions(self):
        return len(self.regions)



    def avg_region_circularity(self):
        try:
            avg_circularity = sum(self.region_circularity()) / len(self.regions)
        except ZeroDivisionError:
            avg_circularity = None
        return avg_circularity

    def perimeter_complexity(self):
        
        try:
            perimeter_complexity = self.total_regions_perimeter() / \
            self.total_regions_area()
        except ZeroDivisionError:
            perimeter_complexity = None
        return perimeter_complexity

   

    def parameters(self):
        """Return a list of filename and all core parameters"""
        params = [self.basename]
        for param in CORE_PARAMETERS:
            if getattr(self, param)() is not None:
                
                try:
                    params.append(round(getattr(self, param)(), 5))
                except TypeError:
                    params.append(None)
            else:
                params.append(getattr(self, param)())
               

        return params
