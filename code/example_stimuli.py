#!/usr/bin/env python3
import os

from image_manipulation import *

if __name__ == "__main__":

    target_dir = "../figures/introduction/"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    use_JPEG = False # either JPEG or PNG
    #img = imload_rgb("test_image.JPEG")
    img = imload_rgb("example_image.png")
    save_img(img, os.path.join(target_dir, "example_image_colour"), use_JPEG)

    ###################################################
    # A) Example for color-experiment:
    #    - convert to grayscale
    ###################################################

    img_grayscale = rgb2gray(img)
    save_img(img_grayscale, os.path.join(target_dir, "example_image_grayscale"), use_JPEG)

    ###################################################
    # B) Example for contrast-experiment:
    #    - convert to grayscale and
    #    - reduce contrast to nominal contrast of 10%
    ###################################################

    contrast_level_1 = 0.3
    img_low_contrast = grayscale_contrast(image=img,
                                          contrast_level=contrast_level_1)
    save_img(img_low_contrast, os.path.join(target_dir, "example_image_low_contrast"), use_JPEG)

    ###################################################
    # C) Example for noise-experiment:
    #    - convert to graycale and
    #    - reduce contrast to 30% and
    #    - apply uniform noise with width 0.1
    ###################################################

    noise_width = 0.2
    contrast_level_2 = 0.3
    rng = np.random.RandomState(seed=42)

    img_noisy = uniform_noise(image=img, width=noise_width,
                              contrast_level=contrast_level_2,
                              rng=rng)
    save_img(img_noisy, os.path.join(target_dir, "example_image_uniform_noise"), use_JPEG)

    ###################################################
    # C) Example for salt-and-pepper noise:
    #    - convert to graycale and
    #    - reduce contrast to 30% and
    #    - apply salt-and-pepper-noise with width 0.1
    ###################################################

    noise_levels = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
    contrast_level_3 = 0.3
    img_noisy = salt_and_pepper_noise(image=img,
                                      p = noise_levels[1],
                                      contrast_level = contrast_level_3,
                                      rng = rng)
    save_img(img_noisy, os.path.join(target_dir, "example_image_salt_and_pepper_noise"), use_JPEG)

    ###################################################
    # E) Example for eidolon-experiment:
    #    - use partially_coherent_disarray 
    ###################################################

    #grain = 10.0
    #coherence = 1.0
    #reach = 8.0

    #img_eidolon = eidolon_partially_coherent_disarray(img, reach,
    #                                                  coherence, grain) 
    #save_img(img_eidolon, "test_image_eidolon", use_JPEG)
    
    ###################################################
    # F) Example for false-colour-experiment: 
    ###################################################

    # load function for monitor non-linearity
    x_gamma_function = np.load('./x_gamma_function.npy')

    img_false_colour = false_colour(img, x_gamma_function) 
    save_img(img_false_colour, os.path.join(target_dir, "example_image_false-colour"), use_JPEG)
    
    ###################################################
    # G) Example for highpass-experiment: 
    #    - use a standard devation of 3
    ###################################################
    
    std = 3

    img_highpass = high_pass_filter(img, std) 
    save_img(img_highpass, os.path.join(target_dir, "example_image_highpass"), use_JPEG)
    
    ###################################################
    # H) Example for lowpass-experiment: 
    #    - use a standard devation of 10
    ###################################################
    
    std = 3

    img_lowpass = low_pass_filter(img, std) 
    save_img(img_lowpass, os.path.join(target_dir, "example_image_lowpass"), use_JPEG)
    
    ###################################################
    # I) Example for phase-scrambling: 
    #    - use a noise width of 90 degrees
    ###################################################
    
    width = 90

    img_phase_scrambling = phase_scrambling(img, width) 
    save_img(img_phase_scrambling, os.path.join(target_dir, "example_image_phase-scrambling"), use_JPEG)
    
    ###################################################
    # J) Example for power equalisation: 
    ###################################################

    # load mean amplitude spectrum over all images
    avg_power_spectrum = np.load('./mean_power_spectrum_grey.npy')

    img_power_equalisation = power_equalisation(img, avg_power_spectrum) 
    save_img(img_power_equalisation, os.path.join(target_dir, "example_image_power_equalisation"), use_JPEG)
    
    ###################################################
    # K) Example for rotation: 
    ###################################################

    img_rotation90 = rotate90(img) 
    save_img(img_rotation90, os.path.join(target_dir, "example_image_rotation90"), use_JPEG)
    
    img_rotation180 = rotate180(img) 
    save_img(img_rotation180, os.path.join(target_dir, "example_image_rotation180"), use_JPEG)
    
    img_rotation270 = rotate270(img) 
    save_img(img_rotation270, os.path.join(target_dir, "example_image_rotation270"), use_JPEG)
