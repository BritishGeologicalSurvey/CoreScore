import os
import warnings
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import mlflow.fastai
from fastai.vision import models
from fastai.vision.transform import get_transforms
from fastai.vision.learner import unet_learner
from fastai.vision.image import open_mask
from fastai.vision.data import get_image_files, SegmentationItemList
from fastai.vision import imagenet_stats, DatasetType
CUDA_LAUNCH_BLOCKING = "1"  # better error reporting
# TODO move to v2 when MLFlow does
#from fastai.vision.all import *
from corescore.masks import LABELS


URI = os.environ.get('MLFLOW_TRACKING_URI', '')
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
        self.reduce_size = np.array([148, 1048])  # TODO derive from samples e.g. src_size / 6

    def image_src(self):
        """Load images from self.path/images"""
        self.src = (
            SegmentationItemList.from_folder(
                self.path_img).split_by_rand_pct().label_from_func(
                    self.get_y_fn, classes=LABELS))  # noqa: E501
        self.src.train.y.create_func = partial(open_mask, div=True)
        self.src.valid.y.create_func = partial(open_mask, div=True)
        return self.src

    def image_data(self):
        """Load scaled-down source images as data"""
        self.data = self.image_src().transform(
            get_transforms(), size=self.reduce_size, tfm_y=True).databunch(
            bs=self.batch_size, num_workers=0).normalize(imagenet_stats)  # hmm
        return self.data

    def acc_rock(self, input, target):
        """Metric to assess label quality
        TODO revise this once we have more labels"""
        target = target.squeeze(1)
        mask = target != LABELS.index("Void")
        return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

    def learner(self):
        """Run the UNet learner based on image data"""
        metrics = self.acc_rock
        self.learn = unet_learner(
            self.image_data(),
            models.resnet34,
            metrics=metrics,
            wd=self.wd)
        self.learn.model = torch.nn.DataParallel(self.learn.model)
        self.learn.lr_find()

    def fit(self, lr=5.20E-05):
        """Fit the model for N epochs (defaults to 10)
        with a learning rate (lr - defaults to %20.E.05
        """
        # TODO fix this to use the model's discovered LR
        if not lr:
            lr = 5.20E-05
        self.learner().fit_one_cycle(self.epochs,
                                     slice(lr),
                                     pct_start=self.pct_start)

    def get_y_fn(self, x):
        """Return a file path to a mask given an image path"""
        return self.path_lbl / f'{x.stem}{x.suffix}'

    def save(self):
        """Save the model"""
        # TODO save via MLFLow
        self.learn.save('model')


if __name__ == '__main__':
    coremodel = CoreModel(os.getcwd())
    coremodel.fit()
    coremodel.save()
