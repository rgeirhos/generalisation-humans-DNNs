#!/usr/bin/env python3

#------------------------------------------------#
#    THIS FILE HAS NOT BEEN TESTED!              #
#------------------------------------------------#

from scipy import fftpack as fp
from PIL import Image
from skimage.io import imread
from skimage import img_as_ubyte
from scipy.misc import toimage
import numpy as np
import os
from skimage.color import rgb2grey

#------------------------------------------------#
#                Functions:                      #
#------------------------------------------------#

def get_amplitude_spectrum(image):
    """Returns a greyscale converted image's power spectrum."""

    channel = rgb2grey(image)

    # Fourier Forward Tranform and shift to centre
    f = fp.fft2(channel)
    f = fp.fftshift(f)

    # get amplitudes and phases
    return np.abs(f)


def get_mean_amplitude_spectrum(img_paths:list, save_path:Optional[str]='./mean_power_spectrum_grey.npy'):
    """Calculate the mean power spectrum over all images.
    
    parameters:
    - img_paths: a list containing the paths to all images.
    - save_path: path to save mean power spectrum to"""
    
    # get shape and dtype of images from first image in list
    image1 = imread(img_paths[0]) / 255
    mean_power_spectrum = np.zeros(image1[:,:,0].shape, image1.dtype)
    num_images = len(img_paths)

    # sum amplitude spectra over images
    for img_path in img_paths:
        image = imread(img_path) / 255.0
        mean_amplitude_spectrum = mean_amplitude_spectrum + (calculate_amplitude_spectrum(image) / num_images) 
    
    # save power/amplitude spectrum
    if not os.path.exists(save_path):
        f = open(save_path)
        f.close()
    np.save(save_path, mean_amplitude_spectrum)
