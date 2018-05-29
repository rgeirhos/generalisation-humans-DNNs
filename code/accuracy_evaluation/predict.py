#!/usr/bin/env python3
"""
Define methods to obtain predictions from a trained network.
Used to evaluate ResNet-50 trained from scratch on distortions.
"""


import tensorflow as tf
import os
import numpy as np
from PIL import Image
from os.path import join

import imagenet_16


def create_estimator(model_dir, model_fn):
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params={
            'resnet_size': 50,
            'data_format': None,
            'batch_size': 100,
            'multi_gpu': False,
            'version': 2,
        })
    return classifier


def create_estimator_16(model_dir):
    return create_estimator(model_dir, imagenet_16.imagenet_model_fn)


def preprocess_images(images):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    mean = np.array([_R_MEAN, _G_MEAN, _B_MEAN], dtype=np.float32)
    images = images - mean
    return images


def predict(model_dir, images):
    # expects images in the range [0, 255]
    assert images.dtype == np.float32
    assert images.shape[1:] == (224, 224, 3)

    classifier = create_estimator_16(model_dir)
    N = len(images)
    images = preprocess_images(images)
    images = {'image': images, 'weight': np.ones((len(images),))}
    input_fn = tf.estimator.inputs.numpy_input_fn(
        images, batch_size=100, shuffle=False)
    results = list(classifier.predict(input_fn))
    classes = np.array([result['classes'] for result in results])
    probabilities = np.stack([result['probabilities'] for result in results])
    logits = np.stack([result['logits'] for result in results])
    assert classes.shape == (N,)
    assert probabilities.shape == (N, 16)
    assert logits.shape == (N, 16)
    # returns predicted class for each image as well as probabilities
    # and logits for each class and image
    return classes, probabilities, logits


def ordered_classnames():
    return ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane',
            'clock', 'oven', 'chair', 'bear', 'boat', 'cat', 'bottle',
            'truck', 'car', 'bird', 'dog']


def load_image(image_path, resize=False):
    image = Image.open(image_path)

    if resize:
        w, h = image.size
        if w < h:
            h = int(h / w * 224)
            w = 224
        else:
            w = int(w / h * 224)
            h = 224
        image = image.resize((w, h))

    # center crop
    width, height = image.size
    new_width = new_height = 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image = image.crop((left, top, right, bottom))

    # as array
    image = np.asarray(image, dtype=np.float32)
    return image


# ########################################################################
# Toy example for evaluating a model on an image
# ########################################################################

def main():

    # model name (change if necessary)
    model = 'sixteen01v4'

    # model directory (update if necessary)
    model_dir = join('./checkpoints', model)

    image = load_image('example.jpg')
    # predict expects a batch of images
    images = image[np.newaxis]
    _, probabilities, _ = predict(model_dir, images)
    probabilities = probabilities[0]

    classnames = ordered_classnames()
    for classname, p in zip(classnames, probabilities):
        print(f'{classname:20}: {p * 100:.1f}%')

    # map from arbitrary ordering to sorted order
    sort_indices = np.argsort(ordered_classnames())
    print(probabilities[sort_indices]) 


if __name__ == '__main__':
    main()
