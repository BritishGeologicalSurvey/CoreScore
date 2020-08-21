import os
from corescore.masks import CoreImageProcessor


def process_images(image_dir, labels):
    coreProcessor = CoreImageProcessor("Images", labels=labels)
    for image in coreProcessor.core_types:
        mask_file = coreProcessor.processImage(image)


if __name__ == '__main__':
    process_images('Images/train', 'Core_labels.json')

