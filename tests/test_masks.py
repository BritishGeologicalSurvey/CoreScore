import os
import pytest
from skimage.io import imread
from corescore.masks import CoreImageProcessor
import numpy as np

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
        'fixtures', 'labels.json')


def test_load_processor(image_dir, labels):
    coreProcessor = CoreImageProcessor(image_dir, labels=labels)
    # TODO store/move labels differently
    assert len(coreProcessor.core_types)
    total = 0
    for image in coreProcessor.core_types:
        mask_file = coreProcessor.processImage(image)
        im_arr = imread(mask_file).flatten()
        has_data = im_arr.any()
        print(np.shape(im_arr))
        # Test if mask is blank
        assert has_data
        assert np.count_nonzero(im_arr==0) == np.shape(im_arr)[0] 
        if has_data: total += 1
    assert total > 1
