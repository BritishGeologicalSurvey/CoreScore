import os
from pathlib import Path
from functools import partial
CUDA_LAUNCH_BLOCKING="1"  # better error reporting
import warnings
from fastai.vision import imagenet_stats, DatasetType
from fastai.vision.data import get_image_files, SegmentationItemList
from fastai.vision.image import open_mask
from fastai.vision.learner import unet_learner
from fastai.vision.transform import get_transforms
from fastai.vision import models
#from fastai.vision.all import *

import mlflow.fastai
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from corescore.masks import LABELS



URI = 'http://hwlc7-vyron.bgslcdevops.test:5000'
mlflow.fastai.autolog()
mlflow.set_tracking_uri(URI)

class CoreModel():

    def __init__(self, path, batch_size=1, wd=1e-2,
                 epochs=10, pct_start=0.9):
        path = Path(path)
        self.path_lbl = path / 'Images/train_masks'
        self.path_img = path / 'Images/train'
        self.batch_size = batch_size
        self.wd = wd
        self.pct_start = pct_start
        self.epochs = epochs
        self.reduce_size = np.array([160, 1048]) # TODO derive from samples

#src_size / 6
#array([ 148., 1048.])
    def image_src(self):
        self.src = (SegmentationItemList.from_folder(self.path_img).split_by_rand_pct().label_from_func(self.get_y_fn, classes=LABELS))
        self.src.train.y.create_func = partial(open_mask, div=True)
        self.src.valid.y.create_func = partial(open_mask, div=True)
        return self.src

    def image_data(self):
        self.data = self.image_src().transform(get_transforms(),
                                               size=self.reduce_size,
                                               tfm_y=True)
        bunch = self.data.databunch(bs=self.batch_size, num_workers=0) # set 0 to avoid ForkingPickler pipe error in Windows
        bunch.normalize(imagenet_stats)

    def learner(self):
        def acc_rock(input, target):
            target = target.squeeze(1)
            mask = target != LABELS.index("Void")
            return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

        metrics = acc_rock
        self.learn = unet_learner(self.image_data(), models.resnet34, metrics=metrics, wd=self.wd)
        self.learn.model = torch.nn.DataParallel(learn.model)
        self.learn.lr_find()

    def fit(self):
        # TODO fix this to use the model's discovered LR
        lr = 5.20E-05
        self.learner().fit_one_cycle(self.epochs,
                                     slice(lr),
                                     pct_start=self.pct_start)

    def get_y_fn(self, x):
        return self.path_lbl/f'{x.stem}{x.suffix}'

    def save(self):
        # TODO save via MLFLow
        self.learn.save('model')


if __name__ == '__main__':
    coremodel = CoreModel(os.getcwd())
    coremodel.fit()
    coremodel.save()

