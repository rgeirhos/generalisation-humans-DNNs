#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
from skimage.io import imread, imshow
from matplotlib.font_manager import FontProperties
from matplotlib import cm, colors
from scipy import ndimage
from skimage.color import rgb2gray

from pydnn.image.io import imload_rgb
import pydnn.stimuli as st
import pydnn.utility as ut
from pydnn.human_data_assignment import grayscale_contrast, uniform_noise
from pydnn.image import image as im
from pylab import *
from pydnn import human_data_assignment as hd

"""Functionality to plot stimuli at a certain level of manipulation."""


human_100 = '#A51E37'



vgg_100 = "#0069AA"
googlenet_100 = "#50AAC8"
alexnet_100 = "#415A8C"

plot_colors = [human_100, vgg_100,
               googlenet_100, alexnet_100]

def plot_stimuli_all_conditions(imglist, stimulus_levels,
                                img_manip_func, multiply_labels_by=1,
                                labels_to_int=True,
                                ylabel=None, filename=None,
                                thresholds=None, set_vmin_vmax=True,
                                rotate_angle=0):
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
                  [human_observer, VGG-16, GoogLeNet, AlexNet]
    """

    num_imgs = len(imglist)
    num_stimuli = len(stimulus_levels)


    offset = 0
    if thresholds is not None:
        assert(len(thresholds) == 4), "len (thresholds) needs to be 4"
        if len(stimulus_levels) is len(thresholds):
            offset = 0.3 # add some more vspace

    fig = plt.figure(figsize=(0.4+2*len(imglist),len(stimulus_levels)*1.6+offset))
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
                    subplot.set_ylabel(str(int(s*multiply_labels_by)))
                else:
                    subplot.set_ylabel(str(s*multiply_labels_by))
            counter += 1

    if ylabel is None:
        ylabel = ""
    fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical')

    #fig.tight_layout()
    #plt.show()
    if filename is None:
        plt.show()    
    else:
        plt.savefig(filename)
    


def plot_stimuli_eidolons(img,
                          param1_list, param2_list, param3,
                          img_manip_func,
                          xlabel=None, ylabel=None,
                          filename=None, print_labels=True):
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
                  [human_observer, VGG-16, GoogLeNet, AlexNet]
    """

    num_rows = len(param1_list)
    num_columns = len(param2_list)

    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(top=0.95, bottom=0.05,
                        right=0.95, left=0.05,
                        hspace=0.05)

    counter = 0
 
    for i, param1 in enumerate(param1_list):
        for j, param2 in enumerate(param2_list):
            subplot = fig.add_subplot(num_rows, num_columns, counter+1)
            plt.imshow(img_manip_func(img, param1, param3, param2), #TODO: order ugly
                       cmap="gray")
            subplot.get_yaxis().set_ticks([])
            subplot.get_xaxis().set_ticks([])
            
            if print_labels: 
                if i is (num_columns - 1):
                    subplot.set_xlabel(str(param2))
                if j is 0:
                    subplot.set_ylabel(str(param1))

            counter += 1
    if print_labels:
        if ylabel is None:
            ylabel = ""
        fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical')

        if xlabel is None:
            xlabel = ""
        fig.text(0.5, 0.01, xlabel, va='center', rotation='horizontal')
 
        # text for main
        fig.text(0.08, 0.97, 
             "Partially coherent disarray for coherence = "+str(param3),
             va='center', rotation='horizontal', fontsize=12)

    if filename is None:
        plt.show()    
    else:
        plt.savefig(filename)
    



def main():
    print("---main executing---")

    im1 = imload_rgb("randomly_selected_imgs/random_bicycle.JPEG")
    im2 = imload_rgb("randomly_selected_imgs/random_dog.JPEG")
    im3 = imload_rgb("randomly_selected_imgs/random_keyboard.JPEG")

    im1_col = imload_rgb("randomly_selected_imgs/n03792782_1155_col.JPEG")
    im2_col = imload_rgb("randomly_selected_imgs/n02099601_634_col.JPEG")
    im3_col = imload_rgb("randomly_selected_imgs/n04505470_10690_col.JPEG")

    im1_2 = imread("randomly_selected_imgs/random_bicycle.JPEG")
    im2_2 = imread("randomly_selected_imgs/random_dog.JPEG")
    im3_2 = imread("randomly_selected_imgs/random_keyboard.JPEG")

    number = 1

    if number is 1: # CONTRAST-EXPERIMENT
        contrast_levels = [1.0, 0.5, 0.3, 0.15, 0.10, 0.05, 0.03, 0.01]
        plot_stimuli_all_conditions([im1, im2, im3],
          contrast_levels, grayscale_contrast, 100, ylabel=
          "Contrast level in percent", 
          filename="contrast_all-conditions.png")

    if number is 2: # NOISE-EXPERIMENT
 
        rng = np.random.RandomState(seed=42)
        u_noise = lambda i, x: uniform_noise(i, x, 0.3, rng)

        noise_levels = [0.0, 0.03, 0.05, 0.1, 0.2, 0.35, 0.6, 0.9]
        plot_stimuli_all_conditions([im1, im2, im3],
                                noise_levels,
                                u_noise, labels_to_int=False,
                                ylabel="Noise width",
                                filename="noise_all-conditions.png")


    if number is 3: # CONTRAST-EXPERIMENT THRESHOLD STIMULI


        # order: human, vgg, googlenet, alexnet
        # contrast_levels = [0.0512, 0.0659, 0.0961, 0.1262] # old (psychometric function)
        contrast_levels = [0.0534, 0.0731, 0.1146, 0.1537]
        plot_stimuli_all_conditions([im1, im2, im3],                                            
          contrast_levels, grayscale_contrast, 100, ylabel=
          "Nominal contrast level in percent", 
          labels_to_int=False,
          filename="../figures/results/contrast/contrast_threshold.png",
          thresholds = contrast_levels)
 

    if number is 3.5: # CONTRAST-PNG-EXPERIMENT THRESHOLD STIMULI

        # order: human, vgg, googlenet, alexnet
        # contrast_levels = [0.0512, 0.0659, 0.0961, 0.1262] # old (psychometric function)
        contrast_levels = [0.0453, 0.0428, 0.0931, 0.1381]
        plot_stimuli_all_conditions([im1, im2, im3],                                            
          contrast_levels, grayscale_contrast, 100, ylabel=
          "Contrast level in percent", 
          labels_to_int=False,
          filename="../figures/results/contrast/contrast_png_threshold.png",
          thresholds = contrast_levels)
 


    if number is 4: # NOISE-EXPERIMENT THRESHOLD STIMULI

        rng = np.random.RandomState(seed=42)
        u_noise = lambda i, x: uniform_noise(i, x, 0.3, rng)

        # order: human, vgg, googlenet, alexnet
        #noise_levels = [0.37, 0.10, 0.09, 0.06] # old (psychometric function)
        noise_levels = [0.307, 0.090, 0.076, 0.051] # mew (accuracy = 0.5)
        plot_stimuli_all_conditions([im1, im2, im3],
               noise_levels, u_noise, labels_to_int=False,
               ylabel="Noise width",
               filename="../figures/results/noise/noise_threshold.png",
               thresholds=noise_levels)

    if number is 5: # EIDOLON-EXPERIMENT coh=1.0 THRESHOLD STIMULI

        funct = lambda i, reach: hd.eidolon_partially_coherent_disarray(i, reach, coherence=1.0, grain=10.0)

        # order: human, vgg, googlenet, alexnet
        reach_levels = [4.001, 2.517, 2.753, 2.808]
        grain = 10.0
        coherence_levels = [1.0]
        coh_in_filename=["10"]

        for i, c in enumerate(coherence_levels):

            eidolon_function = lambda i, reach: hd.eidolon_partially_coherent_disarray(i, 2**reach, c, 10.0)

            plot_stimuli_all_conditions([im1_2, im2_2, im3_2],
                                reach_levels,
                                eidolon_function, labels_to_int=False,
                                ylabel="Log$_2$ of reach parameter",
                                filename=("../figures/results/eidolon/e10_threshold.png"),
                                thresholds=reach_levels, set_vmin_vmax=False)


    if number is 6: # EIDOLON-EXPERIMENT coh=0.0 THRESHOLD STIMULI

        funct = lambda i, reach: hd.eidolon_partially_coherent_disarray(i, reach, coherence=1.0, grain=10.0)

        # order: human, vgg, googlenet, alexnet
        reach_levels = [2.659, 1.934, 2.138, 1.844]
        grain = 10.0
        coherence_levels = [0.0]
        coh_in_filename=["10"]

        for i, c in enumerate(coherence_levels):

            eidolon_function = lambda i, reach: hd.eidolon_partially_coherent_disarray(i, 2**reach, c, 10.0)

            plot_stimuli_all_conditions([im1_2, im2_2, im3_2],
                                reach_levels,
                                eidolon_function, labels_to_int=False,
                                ylabel="Log$_2$ of reach parameter",
                                filename=("../figures/results/eidolon/e0_threshold.png"),
                                thresholds=reach_levels, set_vmin_vmax=False)

    if number is 6.3: # EIDOLON-EXPERIMENT coh=0.3 THRESHOLD STIMULI

        funct = lambda i, reach: hd.eidolon_partially_coherent_disarray(i, reach, coherence=1.0, grain=10.0)

        # order: human, vgg, googlenet, alexnet
        reach_levels = [3.256, 2.318, 2.429, 2.248]
        grain = 10.0
        coherence_levels = [0.3]
        coh_in_filename=["10"]

        for i, c in enumerate(coherence_levels):

            eidolon_function = lambda i, reach: hd.eidolon_partially_coherent_disarray(i, 2**reach, c, 10.0)

            plot_stimuli_all_conditions([im1_2, im2_2, im3_2],
                                reach_levels,
                                eidolon_function, labels_to_int=False,
                                ylabel="Log$_2$ of reach parameter",
                                filename=("../figures/results/eidolon/e3_threshold.png"),
                                thresholds=reach_levels, set_vmin_vmax=False)



    if number is 7: # EIDOLON VISUALIZATION

        funct = lambda i, reach, coherence, grain: hd.eidolon_partially_coherent_disarray(i, reach, coherence, grain)

        reach_levels = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        grain_levels = [1.0, 4.0, 8.0, 16.0, 32.0, 42.0]

        coherence_levels = [0.0, 0.5, 1.0]
        coh_in_filename = ["0.0", "0.5", "1.0"]

        for i, c in enumerate(coherence_levels):

            plot_stimuli_eidolons(im4,
               reach_levels,
               grain_levels,
               c,
               funct,
               ylabel="Reach",
               xlabel="Grain",
               filename=("/home/robert/eidolon_"+
                         "grid_labels_coherence-"+
                         coh_in_filename[i]+
                         ".png"),
               print_labels=True)
               #filename=None)
 
            print("Finished run #"+str(i+1))

    if number is 8: # EIDOLON-EXPERIMENT
 
        grain = 10.0
        coherence_levels = [0.0, 0.3, 1.0]
        coh_in_filename=["00", "03", "10"]
        reach_levels = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

        for i, c in enumerate(coherence_levels):

            eidolon_function = lambda i, reach: hd.eidolon_partially_coherent_disarray(i, reach, c, 10.0)

            plot_stimuli_all_conditions([im1_2, im2_2, im3_2],
                                reach_levels,
                                eidolon_function, labels_to_int=False,
                                ylabel="Reach level",
                                filename=("eidolon_coh="+
                                          coh_in_filename[i]+"_all-conditions.png"),
                                thresholds=None, set_vmin_vmax=False)

    if number is 9: # ALPHA-NOISE-EXPERIMENT
 
        rng = np.random.RandomState(seed=42)

        def alpha_noise(image, alpha, rng):
            return st.apply_alpha_noise(rgb2gray(image), alpha, rng)

        a_noise = lambda x, a, rng=rng: alpha_noise(x, a, rng)

        alpha_levels = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        plot_stimuli_all_conditions([im1, im2, im3],
                                alpha_levels,
                                a_noise, labels_to_int=False,
                                ylabel="Noise alpha",
                                filename="alpha_noise_all-conditions.png")


    if number is 10: # CONTRAST-EXPERIMENT FOR POSTER
        contrast_levels = [1.0, 0.5, 0.3, 0.15, 0.10, 0.05, 0.03, 0.01]
        plot_stimuli_all_conditions([im1, im2, im3],
          contrast_levels, grayscale_contrast, 100, ylabel=
          "Contrast level in percent", 
          filename="../figures/poster/contrast_all-conditions-poster.png",
          rotate_angle=90)


    if number is 11: # NOISE-EXPERIMENT FOR POSTER
 
        rng = np.random.RandomState(seed=42)
        u_noise = lambda i, x: uniform_noise(i, x, 0.3, rng)

        noise_levels = [0.9, 0.6, 0.35, 0.2, 0.1, 0.05, 0.03, 0.0]
        plot_stimuli_all_conditions([im1, im2, im3],
                                noise_levels,
                                u_noise, labels_to_int=False,
                                ylabel="Noise width",
                                filename="../figures/poster/noise_all-conditions-poster.png",
                                rotate_angle=90)

    if number is 12: # EIDOLON-EXPERIMENT FOR POSTER (FULL COHERENCE)
 
        grain = 10.0
        coherence_levels = [1.0]
        coh_in_filename=["10"]
        reach_levels = [7, 6, 5, 4, 3, 2, 1, 0]

        for i, c in enumerate(coherence_levels):

            eidolon_function = lambda i, reach: hd.eidolon_partially_coherent_disarray(i, 2**reach, c, 10.0)

            plot_stimuli_all_conditions([im1_2, im2_2, im3_2],
                                reach_levels,
                                eidolon_function, labels_to_int=True,
                                ylabel="Log$_2$ of reach parameter",
                                filename=("../figures/poster/eidolon_coh="+
                                          coh_in_filename[i]+"_all-conditions-poster.png"),
                                thresholds=None, set_vmin_vmax=False, rotate_angle=90)

    if number is 13: # COLOR-EXPERIMENT FOR POSTER
 

        color_fun = lambda i, l: i if l is "Color" else rgb2gray(i)

        color_levels = ["Grayscale", "Color"]
        plot_stimuli_all_conditions([im1_col, im2_col, im3_col],
                                color_levels,
                                color_fun, labels_to_int=False,
                                ylabel="Color vs. grayscale",
                                filename="../figures/poster/color-all_conditions-poster.png",
                                rotate_angle=90)



def misc():

    nrow, ncol = im2.shape


    fig = plt.figure()
    contrast = 0.3
    noise_width = 0.5

    noise1 = st.get_uniform_noise(-noise_width, noise_width,
                                  nrow, ncol)

    im2 = st.adjust_contrast(im2, contrast)

    a1 = fig.add_subplot(3,3,1)
    plt.imshow(im2, cmap="gray", vmin=0.0, vmax=1.0)
    a1.set_title("original image (contrast="+ str( contrast) + ")")

    a2 = fig.add_subplot(3,3,2)
    plt.imshow(st.fourier_transform_image(im2), cmap="gray")
    a2.set_title("original image: FT (amplitude)")

    a3 = fig.add_subplot(3,3,3)
    plt.hist(im2.ravel(), bins=256, fc="k", ec="k")
    a3.set_title("original image: histogram")


    b1 = fig.add_subplot(3,3,4)
    plt.imshow(noise1+0.5, cmap="gray", vmin=0.0, vmax=1.0)
    b1.set_title("0.5+noise (noise_width="+str(noise_width)+")")
  
    b2 = fig.add_subplot(3,3,5)
    plt.imshow(st.fourier_transform_image(noise1+0.5), cmap="gray")
    b2.set_title("noise: FT (amplitude)")

    b3 = fig.add_subplot(3,3,6)
    plt.hist(noise1.ravel(), bins=256, range=(-1.0, 1.0), fc="k", ec="k")
    b3.set_title("noise: histogram")

    noisy_image = im2 + noise1
    noisy_image = np.where(noisy_image < 0, 0, noisy_image)
    noisy_image = np.where(noisy_image > 1, 1, noisy_image)
    assert ut.is_in_bounds(noisy_image, 0, 1), "ERROR - values outside occurred."

    c1 = fig.add_subplot(3,3,7)
    plt.imshow(noisy_image, cmap="gray", vmin=0.0, vmax=1.0)
    c1.set_title("image + noise")

    c2 = fig.add_subplot(3,3,8)
    plt.imshow(st.fourier_transform_image(noisy_image), cmap="gray")
    c2.set_title("image + noise: FT (amplitude)")

    c3 = fig.add_subplot(3,3,9)
    plt.hist(noisy_image.ravel(), bins=256, fc="k", ec="k")
    c3.set_title("image + noise: histogram")


    plt.show()




    """
    #a = st.load_image("example_image.jpeg")
    #a.show()
    #print(a)

    #print(st.load_all_images("images/"))

    #image_list = st.load_all_images("/home/robert/vision-model-DNN/code/psycho_dnn/images/")
    #image_list = st.load_all_images("images/")
    #print("length of list: "+str(len(image_list)))

    im = (next(image_list))
    im = st.convert_to_grayscale(im)    

    im2 = st.convert_image_to_0_1_array(im)

    print(np.amax(im2))
    print(np.amin(im2))   
   
 
    im3 = Image.fromarray(im2, mode="L")
    im3.show()

    rng = np.random.RandomState(seed=444)

    #all_images = st.manipulate_grayscale_contrast_noise_all_images("images/", 0.6, 50, rng)

    #curr_image = next(all_images)
    #curr_image.show()

    #curr_image = next(all_images)
    #curr_image.show()
    """
    
    noise_vec = np.linspace(0.0, 0.25, 9)
    #noise_vec = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.25, 0.25, 0.25]
    show_different_noise_sigmas(im2, noise_vec,
                                contrast=0.2, max_noise_sd=1.5)

    #show_with_histograms(im2, noise_sigma=0.25, contrast=0.2, max_noise_sd=1.5)


    print("---main done---")




def show_different_noise_sigmas(image, noise_vec, contrast, max_noise_sd):

    nrow, ncol = image.shape
    fig = plt.figure()

    contrast_reduced_image = st.adjust_contrast(image, contrast)

    for i in range(9):

        noise = st.get_truncated_noise(nrow, ncol, max_noise_sd, noise_vec[i])
        noisy_image = contrast_reduced_image + noise
 
        if not ut.is_in_bounds(noisy_image, 0, 1):
            print("values <0 or >1 occurred")

        a = fig.add_subplot(3, 3, (i+1))
        plt.imshow(noisy_image, cmap="gray", vmin=0.0, vmax=1.0)
        a.set_title("noise_sigma: "+str(noise_vec[i]))
 
 
    plt.show()



def show_with_histograms(image, noise_sigma, contrast, max_noise_sd):
    """Show image, noise, and noisy image with FT and histogram."""

    nrow, ncol = image.shape
    fig = plt.figure()

    noise1 = st.get_truncated_noise(nrow, ncol, max_noise_sd, noise_sigma)
    image = st.adjust_contrast(image, contrast)

    a1 = fig.add_subplot(3,3,1)
    plt.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    a1.set_title("original image (contrast="+ str( contrast) + ")")

    a2 = fig.add_subplot(3,3,2)
    plt.imshow(st.fourier_transform_image(image), cmap="gray")
    a2.set_title("original image: FT (amplitude)")

    a3 = fig.add_subplot(3,3,3)
    plt.hist(image.ravel(), bins=256, fc="k", ec="k")
    a3.set_title("original image: histogram")


    b1 = fig.add_subplot(3,3,4)
    plt.imshow(noise1+0.5, cmap="gray", vmin=0.0, vmax=1.0)
    b1.set_title("0.5+noise (MAX SD=1.5, noise_sigma="+str(noise_sigma)+")")
  
    b2 = fig.add_subplot(3,3,5)
    plt.imshow(st.fourier_transform_image(noise1+0.5), cmap="gray")
    b2.set_title("noise: FT (amplitude)")

    b3 = fig.add_subplot(3,3,6)
    plt.hist(noise1.ravel(), bins=256, range=(-1.0, 1.0), fc="k", ec="k")
    b3.set_title("noise: histogram")

    noisy_image = image + noise1
    c1 = fig.add_subplot(3,3,7)
    plt.imshow(noisy_image, cmap="gray", vmin=0.0, vmax=1.0)
    c1.set_title("image + noise")

    c2 = fig.add_subplot(3,3,8)
    plt.imshow(st.fourier_transform_image(noisy_image), cmap="gray")
    c2.set_title("image + noise: FT (amplitude)")

    c3 = fig.add_subplot(3,3,9)
    plt.hist(noisy_image.ravel(), bins=256, range=(0.0, 1.0), fc="k", ec="k")
    c3.set_title("image + noise: histogram")

    plt.show()


def test(i):
    print(43)


if __name__ == "__main__":
    main()
