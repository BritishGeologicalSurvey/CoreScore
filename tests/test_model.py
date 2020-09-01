import os
import fastai
import pytest
from corescore.models import CoreModel
from corescore.masks import CoreImageProcessor


@pytest.fixture
def image_dir():
    fix_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'fixtures')
    return fix_dir


@pytest.fixture
def make_masks(image_dir):
    coreProcessor = CoreImageProcessor(image_dir+'Images', labels=labels)
    for image in coreProcessor.core_types:
        coreProcessor.processImage(image)

def test_create_model():
    model = CoreModel(os.getcwd())


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
