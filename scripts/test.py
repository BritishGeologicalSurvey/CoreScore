# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:50:52 2021

@author: Alex
"""

from corescore.api import load_model
from fastai.vision import *


TEST_DIR = './Images/test'
MODEL_NAME = 'tmp'
PRED_DIR = './Images/test_predictions'

unet = load_model.load_corescore_model(MODEL_NAME)  # TODO, does not work, load_model changed and has not been updated


if __name__ == "_main__":

    for im in os.listdir(TEST_DIR):
        
        img = open_image(os.path.join(TEST_DIR, im))
        pred = unet.predict(img)[0]
        pred.show()
        plt.savefig(os.path.join(PRED_DIR, im))
