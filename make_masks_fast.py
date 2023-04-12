# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 23:41:51 2021

@author: Alex
"""


import argparse
import os
from corescore.masks import CoreImageProcessor


def process_images(image_dir, label):
    """Create masks for labelled images.
    For now, merge rock fragment labels from both core boxes"""
    coreProcessor = CoreImageProcessor("Images",
                                       labels=label,
                                       merge_fragment_labels=True)

    image = coreProcessor.core_types
    mask_file = coreProcessor.processImage(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help="Directory of images to make masks for")
    parser.add_argument('--labels_dir', help="folder of JSON files with labels for images")
    args = parser.parse_args()

    images = args.images
    if not images:
        images = './Images'
    labels_dir = args.labels_dir
    if not labels_dir:
        labels_dir = 'train_labels'
    
    for f in os.listdir(labels_dir):
        
        process_images(images, os.path.join(labels_dir,f))
