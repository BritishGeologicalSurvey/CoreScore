import os
import pytest
from skimage.io import imread
from corescore.masks import CoreImageProcessor


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


@pytest.fixture
def labeltool_labels():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'fixtures', 'labeltool.json')

def test_load_processor(image_dir, labels):
    coreProcessor = CoreImageProcessor(image_dir, labels=labels)
    assert len(coreProcessor.core_types)
    total = 0
    for image in coreProcessor.core_types:
        mask_file = coreProcessor.processImage(image)
        im_arr = imread(mask_file).flatten()
        has_data = im_arr.any()
        # Test if mask is blank
        assert has_data
        if has_data:
            total += 1
    assert total > 1


def test_load_labeltool(image_dir, labeltool_labels):
    coreProcessor = CoreImageProcessor(image_dir, labels=labeltool_labels)
    assert len(coreProcessor.core_types)
    print(coreProcessor.core_types)

    total = 0
    for image in coreProcessor.core_types:
        mask_file = coreProcessor.processImage(image)
        im_arr = imread(mask_file).flatten()
        has_data = im_arr.any()
        # Test if mask is blank
        assert has_data
        if has_data:
            total += 1
    assert total > 1
