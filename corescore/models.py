import os
from pathlib import Path
from functools import partial
CUDA_LAUNCH_BLOCKING="1"  # better error reporting
import warnings
#from fastai.vision import imagenet_stats, DatasetType
#from fastai.vision.data import get_image_files, SegmentationItemList
#from fastai.vision.image import open_mask
#from fastai.vision.learner import unet_learner
#from fastai.vision.transform import get_transforms
#from fastai.vision import models
from fastai.vision.all import *

import mlflow.fastai
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from corescore.masks import LABELS


mlflow.fastai.autolog()

class CoreModel():
    def __init__(self, path):
        path = Path(path)
        self.path_lbl = path / 'Images/train_masks'
        self.path_img = path / 'Images/train'

#src_size / 6
#array([ 148., 1048.])
    def image_src(self):
        self.src = (SegmentationItemList.from_folder(path_img).split_by_rand_pct().label_from_func(get_y_fn, classes=LABELS))
        self.src.train.y.create_func = partial(open_mask, div=True)
        self.src.valid.y.create_func = partial(open_mask, div=True)
        return self.src

    def image_data(self):
        self.data = self.image_src().transform(get_transforms(),
                                               size=reduce_size,
                                               tfm_y=True)
        self.data.databunch(bs=batch_size, num_workers=0) # set 0 to avoid ForkingPickler pipe error in Windows
        self.data.normalize(imagenet_stats)

    def learner(self):
        self.learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
        self.learn.model = torch.nn.DataParallel(learn.model)
        self.learn.lr_find()
