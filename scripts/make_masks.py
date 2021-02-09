import argparse
import os
from corescore.masks import CoreImageProcessor


def process_images(image_dir, labels):
    """Create masks for labelled images.
    For now, merge rock fragment labels from both core boxes"""
    coreProcessor = CoreImageProcessor("Images",
                                        labels=labels,
                                        merge_fragment_labels=True)
    for image in coreProcessor.core_types:
        mask_file = coreProcessor.processImage(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help="Directory of images to make masks for")
    parser.add_argument('--labels', help="JSON file with labels for images")
    args = parser.parse_args()

    images = args.images
    if not images:
        images = 'Images/train'
    labels = args.labels
    if not labels:
        labels = 'Core_labels.json'

    process_images(images, labels)
