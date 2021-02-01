import os
import pytest
from skimage.io import imread
from corescore.masks import CoreImageProcessor
from corescore.quality import QualityIndex


@pytest.fixture
def image_dir():
    fix_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'fixtures', 'images')
    return fix_dir


@pytest.fixture
def labels():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'fixtures', 'labeltool.json')


def test_load_processor(image_dir, labels):
    coreProcessor = CoreImageProcessor(image_dir, labels=labels)
    for image in coreProcessor.core_types:
        mask_file = coreProcessor.processImage(image)
        im_arr = imread(mask_file)
        measures = QualityIndex(im_arr)
        params = measures.parameters()
        assert params
        assert 'total_fragments' in params
