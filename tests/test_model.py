import os
import pytest
import fastai
from fastai.vision.image import Image
from skimage.io import imread, imsave
from torch import Tensor
from torchvision.transforms import ToTensor
from corescore.models import CoreModel
from corescore.masks import LABELS


@pytest.fixture
def image_tensor():
    sample = 'S00128822.Cropped_Top_2_Countoured.png'
    fix_dir = os.path.dirname(os.path.abspath(__file__))
    image_arr = imread(os.path.join(fix_dir,
                                    'fixtures',
                                    'images',
                                    'test',
                                    sample))
    # FastAI v2 should not require decanting the ndarray like this
    return Image(ToTensor()(image_arr))


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


def test_fit_one(image_tensor):
    model = CoreModel(os.getcwd(), epochs=1)
    learn = model.learner()
    model.fit(learn)
    # test a prediction
    _, mask, _ = learn.predict(image_tensor)
    assert isinstance(mask, Tensor)

    img_arr = mask.numpy()[0].astype('uint8')
    # Check the values are in the expected range for labels
    assert (img_arr > 0).any()
    assert (img_arr < len(LABELS)).all()

    # tmp - save mask for inspection / testing
    imsave('test_mask.png', img_arr)
