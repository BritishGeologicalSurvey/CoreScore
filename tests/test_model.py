import os
import fastai
from corescore.models import CoreModel


def test_create_model():
    model = CoreModel(os.getcwd())
    assert model


def test_fastai_version():
    major = fastai.__version__.split('.')[0]
    assert int(major) == 1


def test_image_src():
    model = CoreModel(os.getcwd())
    src = model.image_src()
    assert src


def test_image_data():
    model = CoreModel(os.getcwd())
    data = model.image_data()
    assert data


def test_fit_one():
    model = CoreModel(os.getcwd(), epochs=1)
    model.fit()
