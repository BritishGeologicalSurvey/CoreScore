import os
import fastai
from corescore.models import CoreModel


def test_create_model():
    model = CoreModel(os.getcwd())


def test_version():
    major = fastai.__version__.split('.')[0]
    assert int(major) == 1
