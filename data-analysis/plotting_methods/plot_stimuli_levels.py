#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
from skimage.io import imread, imshow
from matplotlib.font_manager import FontProperties
from matplotlib import cm, colors
from scipy import ndimage
from skimage.color import rgb2grey
from pylab import *

# import manipulation functions
import sys
sys.path.append('../../code')
from image_manipulation import (high_pass_filter, low_pass_filter,
                                phase_scrambling, salt_and_pepper_noise,
                                false_colour, power_equalisation,
                                grayscale_contrast, uniform_noise,
                                eidolon_partially_coherent_disarray)


"""Functionality to plot stimuli at a certain level of manipulation."""


humans = '##A51E37'
googlenet = "#50AAC8"
vgg19 = "#0069AA"
resnet152 = "#415A8C"

plot_colors = [humans, googlenet, vgg19, resnet152]


def plot_stimuli_all_conditions(imglist, stimulus_levels,
                                img_manip_func, multiply_labels_by=1,
                                labels_to_int=True,
                                ylabel=None, filename=None,
                                thresholds=None, set_vmin_vmax=True,
                                rotate_angle=0,
                                reduced_space=False,
                                is_eidolon=False):
    """Plot a nxm matrix of n stimulus_levels for m images.

    parameters:
    - imglist: vector of images
    - stimulus_levels: numeric vector of input values to img_manip_func
    - img_manip_func: Function(image, stimulus_level) -> manipulated img
    - labels_to_int: multiply labels by 100 and convert them to int?
    - ylabel: plot y-axis-label
    - filename: where the resulting plot should be saved
    - thresholds: optional list of stimulus levels that should be plotted in
                  color. The order is important:
                  [human_observer, GoogLeNet, VGG-19, ResNet-152]
    - reduced_space: reduce plotting area for paper
    """

    num_imgs = len(imglist)
    num_stimuli = len(stimulus_levels)


    offset = 0
    if thresholds is not None:
        assert(len(thresholds) == 4), "len (thresholds) needs to be 4"
        if len(stimulus_levels) is len(thresholds):
            offset = 0.3 # add some more vspace

    fig = plt.figure(figsize=(-0.2+2*len(imglist),len(stimulus_levels)*1.6+offset))
    fig.subplots_adjust(top=0.98, bottom=0.02,
                        right=0.93, left=0.09,
                        hspace=0.05)

    counter = 0


    def plot_rectangle(color, subplot):

        autoAxis = subplot.axis()
        rec = Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=5, color=color)
        rec = subplot.add_patch(rec)
        rec.set_clip_on(False)

 
    for stimulus_counter, s in enumerate(stimulus_levels):
        for i, img in enumerate(imglist):
            subplot = fig.add_subplot(num_stimuli,num_imgs, counter+1)
            
            
            ######break inserted here
            
            if not is_eidolon:
                # clip to 0..1 range
                assert np.allclose(img[img < 0], 0) and np.allclose(img[img > 1], 1)
                img[img < 0] = 0
                img[img > 1] = 1
                
            if set_vmin_vmax:
                plt.imshow(ndimage.rotate(img_manip_func(img, s), rotate_angle),
                       cmap="gray", vmin=0.0, vmax=1.0)
            else:
                plt.imshow(ndimage.rotate(img_manip_func(img, s), rotate_angle),
                       cmap="gray")
            subplot.get_yaxis().set_ticks([])
            subplot.get_xaxis().set_ticks([])

            if thresholds is not None:
                if s in thresholds:
                    plot_rectangle(plot_colors[stimulus_counter], subplot)
 
            if i is 0:
                if labels_to_int:
                    subplot.set_ylabel(str(int(s*multiply_labels_by)),
                                       fontsize=12)#, fontsize=16)
                else:
                    subplot.set_ylabel(str(s*multiply_labels_by), fontsize=12)#, fontsize=16)
            counter += 1

    if ylabel is None:
        ylabel = ""
    fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=12)#, fontsize=16)

    #fig.tight_layout() 
    #plt.show()
    if filename is None:
        plt.show()    
    else:
        plt.savefig(filename)


def main(number):
    print("---main executing---")

    im1_col = imread("../randomly_selected_imgs/n03792782_1155_224x224.JPEG") / 255.0
    im2_col = imread("../randomly_selected_imgs/n02099601_634_224x224.JPEG") /  255.0
    im3_col = imread("../randomly_selected_imgs/n04505470_10690_224x224.JPEG") / 255.0

    im1 = rgb2grey(im1_col)
    im2 = rgb2grey(im2_col)
    im3 = rgb2grey(im3_col)

    im1_gray = imread("../randomly_selected_imgs/random_bicycle.JPEG")
    im2_gray = imread("../randomly_selected_imgs/random_dog.JPEG")
    im3_gray = imread("../randomly_selected_imgs/random_keyboard.JPEG")


    # import npy files
    x_gamma_function = np.load('../../code/x_gamma_function.npy')
    avg_power_spectrum = np.load('../../code/mean_power_spectrum_grey.npy')


    if number is 1: # uniform noise experiment
        
        rng = np.random.RandomState(seed=42)
        noise_levels = [0.0, 0.03, 0.05, 0.1, 0.2, 0.35, 0.6, 0.9]

        u_noise = lambda i, x: uniform_noise(i, x, 0.3, rng)

        plot_stimuli_all_conditions([im1, im2, im3],
                                    noise_levels,
                                    u_noise,
                                    labels_to_int=False,
                                    ylabel= "Uniform noise width",
                                    filename="../../figures/methods/noise_all-conditions.png")



    if number is 2: # contrast experiment
        
        contrast_levels = [1.0, 0.5, 0.3, 0.15, 0.10, 0.05, 0.03, 0.01]

        plot_stimuli_all_conditions([im1, im2, im3],
                                    contrast_levels,
                                    grayscale_contrast,
                                    labels_to_int=True,
                                    multiply_labels_by=100,
                                    ylabel= "Contrast level in percent",
                                    filename="../../figures/methods/contrast_all-conditions.png")

    if number is 3: # Eidolon experiments
        
        grain = 10.0
        coherence_levels = [0.0, 0.3, 1.0]
        coh_in_filename = ["00", "03", "10"]
        reach_levels = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

        for i, c in enumerate(coherence_levels):

            eidolon_function = lambda i, reach: eidolon_partially_coherent_disarray(i, reach, c, grain)


            plot_stimuli_all_conditions([im1_gray, im2_gray, im3_gray],
                                        reach_levels,
                                        eidolon_function,
                                        ylabel= "Reach level",
                                        filename=("../../figures/methods/eidolon_coh="+
                                                  coh_in_filename[i]+"_all-conditions.png"),
                                        thresholds=None, set_vmin_vmax=False, is_eidolon=True)



    if number is 4: # opponent colour experiment
        colour = ["true", "opponent"]

        def opponent_colours(img, colour):
            if colour == "opponent":
                return false_colour(img, x_gamma_function)
            else:
                return img

        plot_stimuli_all_conditions([im1_col, im2_col, im3_col],
                                    colour,
                                    opponent_colours,
                                    labels_to_int=False,
                                    ylabel= "Colour",
                                    filename="../../figures/methods/false-colour_all-conditions.png")

    if number is 5: # power equalisation experiment
        power = ["original", "equalised"]

        def equalise_power(img, power):
            if power == "equalised":
                return power_equalisation(img, avg_power_spectrum)
            else:
                return img

        plot_stimuli_all_conditions([im1, im2, im3],
                                    power,
                                    equalise_power,
                                    labels_to_int=False,
                                    ylabel= "Power spectrum",
                                    filename="../../figures/methods/power-equalisation_all-conditions.png")


    if number is 6: # Highpass experiment
        standard_deviations = ["inf", 3, 1.5, 1, 0.7, 0.55, 0.45, 0.4]
        def high_pass_filter_with_inf(img, sd):
            if sd == "inf":
                return img
            else:
                return high_pass_filter(img, sd)

        plot_stimuli_all_conditions([im1, im2, im3],
                                    standard_deviations,
                                    high_pass_filter_with_inf,
                                    labels_to_int=False,
                                    ylabel= "Filter standard deviation [pixels]",
                                    filename="../../figures/methods/highpass_all-conditions_paper.png",
                                    reduced_space = False)

    if number is 7: # Lowpass experiment
        standard_deviations = [0, 1, 3, 5, 7, 10, 15, 40]
        plot_stimuli_all_conditions([im1, im2, im3],
                                    standard_deviations,
                                    low_pass_filter,
                                    labels_to_int=True,
                                    ylabel= "Filter standard deviation [pixels]",
                                    filename="../../figures/methods/lowpass_all-conditions_paper.png",
                                    reduced_space = False)

    if number is 8: # Phase noise experiment
        standard_deviations = [0, 30, 60, 90, 120, 150, 180]
        plot_stimuli_all_conditions([im1, im2, im3],
                                    standard_deviations,
                                    phase_scrambling,
                                    labels_to_int=True,
                                    ylabel= "Phase noise width [deg]",
                                    filename="../../figures/methods/phase-noise_all-conditions_paper.png",
                                    reduced_space = False)


    if number is 9: # Salt-and-pepper experiment
        noise_levels = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
        contrast_level_3 = 0.3
        rng = np.random.RandomState(seed=42)
        def salt_and_pepper_noise_worker(img, p):
            return salt_and_pepper_noise(img, p=p,
                                         contrast_level=contrast_level_3,
                                         rng=rng)
        plot_stimuli_all_conditions([im1, im2, im3],
                                    noise_levels,
                                    salt_and_pepper_noise_worker,
                                    labels_to_int=True,
                                    multiply_labels_by=100,
                                    ylabel= "Percent noise pixels [%]",
                                    filename="../../figures/methods/salt-and-pepper_all-conditions_paper.png",
                                    reduced_space = False)




if __name__ == "__main__":

    # non-Eidolon stimuli: uncomment and execute with Python 3.5
    #for number in [1,2,4,5,6,7,8,9]:
    #    main(number)
        
    # eidolon stimuli: uncomment and execute with Python 2.7
    #for number in [3]:
    #    main(number)
    
    print("Please choose between plotting Eidolon stimuli (Python 2.7) and other stimuli (Python 3.5) by uncommenting the respective functions in the main method.")