import os
import pytest
import fastai
from fastai.vision.image import Image
from skimage.io import imread, imsave
from torch import Tensor, randn, randint
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


@pytest.mark.parametrize("pred,target,expected", [
    (randn((128, 2, 224, 224)), randint(0, 5, (128, 1, 224, 224)), 1 / 5),
    (randn((128, 8, 224, 224)), randint(0, 10, (128, 1, 224, 224)), 1 / 10)])
def test_acc_rock(pred, target, expected):
    model = CoreModel(os.getcwd())
    acc_rock = model.acc_rock(pred, target)
    assert acc_rock == pytest.approx(expected, abs=1e-01)


def test_fastai_version():
    """Check this is v1 (needed for mlflow support)"""
    major = fastai.__version__.split('.')[0]
    assert int(major) == 1


def test_image_src():
    """Test we can load images with test/validation split"""
    model = CoreModel(os.getcwd())
    src = model.image_src()
    assert src


def test_resize():
    """Test we are resizing inputs based on default/supplied size
    """
    model = CoreModel(os.getcwd())
    src = model.image_src()
    # This is a LabelList with train / valid beneath x and y attributes
    sample = src.x.get(0)
    orig_size = sample.size
    new_size = model.image_resize(sample)
    assert int(orig_size[0] / 4) == new_size[0]

    # Check we can overrode the value
    new_size = model.image_resize(sample, resize=2)
    assert int(orig_size[0] / 2) == new_size[0]


def test_image_data():
    """Test the transforms being applied to the inputs"""
    model = CoreModel(os.getcwd())
    data = model.image_data()
    assert data


def test_fit_one(image_tensor):
    """Very short training run and test prediction values"""
    model = CoreModel(os.getcwd(), epochs=1)
    learn = model.learner(resize=12)
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
