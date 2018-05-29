#!/usr/bin/env python

from skimage.color import rgb2gray, rgb2grey
from scipy.ndimage.filters import gaussian_filter
from scipy import fftpack as fp
from skimage.io import imread, imsave
from scipy.misc import toimage
import numpy as np
import sys

###########################################################
#   IMAGE IO
###########################################################

def imload_rgb(path):
    """Load and return an RGB image in the range [0, 1]."""

    return imread(path) / 255.0


def save_img(image, imgname, use_JPEG=False):
    """Save image as either .jpeg or .png"""

    if use_JPEG:
        imsave(imgname+".JPEG", image) 
    else:
        toimage(image,
                cmin=0.0, cmax=1.0).save(imgname+".png")


###########################################################
#   IMAGE MANIPULATION
#
# in general, images are handled as follows:
# - datatype: numpy.ndarray
# - numpy dtype: float64
# - channels: RGB (3 channels)
# - range: [0, 1]
#
###########################################################


def adjust_contrast(image, contrast_level):
    """Return the image scaled to a certain contrast level in [0, 1].

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    assert(contrast_level >= 0.0), "contrast_level too low."
    assert(contrast_level <= 1.0), "contrast_level too high."

    return (1-contrast_level)/2.0 + image.dot(contrast_level)


def grayscale_contrast(image, contrast_level):
    """Convert to grayscale. Adjust contrast.

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    return adjust_contrast(rgb2gray(image), contrast_level)


def uniform_noise(image, width, contrast_level, rng):
    """Convert to grayscale. Adjust contrast. Apply uniform noise.

    parameters:
    - image: a numpy.ndarray 
    - width: a scalar indicating width of additive uniform noise
             -> then noise will be in range [-width, width]
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    image = grayscale_contrast(image, contrast_level)

    return apply_uniform_noise(image, -width, width, rng)


def salt_and_pepper_noise(image, p, contrast_level, rng):
    """Convert to grayscale. Adjust contrast. Apply salt and pepper noise.
    parameters:
    - image: a numpy.ndarray
    - p: a scalar indicating probability of white and black pixels, in [0, 1]
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    assert 0 <= p <= 1

    image = grayscale_contrast(image, contrast_level)
    assert image.ndim == 2

    u = rng.uniform(size=image.shape)

    salt = (u >= 1 - p / 2).astype(image.dtype)
    pepper = -(u < p / 2).astype(image.dtype)

    image = image + salt + pepper
    image = np.clip(image, 0, 1)

    assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"

    return image


def false_colour(image, x_gamma_function):#=np.linspace(0, 1, 256)):
    """Return image converted to have opponent colours.
    
    parameters:
    - image: a numpy.ndarray
    - x_gamma_function: an array mapping colour intensities in [0,255]
                        to measured monitor output"""
    
    # adjust image to [0,1] scale and monitor non-linearity
    image_nl = monitor_nonlinearity(image, x_gamma_function)
    
    # calculate the opponent colours
    new_image = get_opponent_colours(image_nl)

    # readjust to monitor nonlinearity
    new_image = inv_monitor_nonlinearity(new_image, x_gamma_function)

    return new_image


def high_pass_filter(image, std):
    """Apply a Gaussian high pass filter to a greyscale converted image.
    by calculating Highpass(image) = image - Lowpass(image).
    
    parameters:
    - image: a numpy.ndarray
    - std: a scalar providing the Gaussian low-pass filter's standard deviation"""

    # set this to mean pixel value over all images
    bg_grey = 0.4423
    
    # convert image to greyscale and define variable prepare new image
    image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # aplly the gaussian filter and subtract from the original image
    gauss_filter = gaussian_filter(image, std, mode ='constant', cval=bg_grey)
    new_image = image - gauss_filter

    # add mean of old image to retain image statistics
    mean_diff = bg_grey - np.mean(new_image, axis=(0,1))
    new_image = new_image + mean_diff

    # crop too small and too large values
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    return np.dstack((new_image,new_image,new_image))


def low_pass_filter(image, std):
    """Aplly a Gaussian low-pass filter to an image.
    
    parameters:
    - image: a numpy.ndarray
    - std: a scalar providing the Gaussian low-pass filter's standard deviation
    """
    # set this to mean pixel value over all images
    bg_grey = 0.4423
    
    # covert image to greyscale and define variable prepare new image
    image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # aplly Gaussian low-pass filter
    new_image = gaussian_filter(image, std, mode ='constant', cval=bg_grey)

    # crop too small and too large values
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    return np.dstack((new_image,new_image,new_image))


def phase_scrambling(image, width):
    """Apply random shifts to an images' frequencies' phases in the Fourier domain.
    
    parameter:
    - image: an numpy.ndaray
    - width: maximal width of the random phase shifts"""
    
    return scramble_phases(image, width)


def power_equalisation(image, avg_power_spectrum):
    """Equalise images' power spectrum by setting an image's amplitudes 
    in the Fourier domain to the amplitude average over all used images.
    
    parameter:
    - image: a numpy.ndarray"""
    
    return equalise_power_spectrum(image, avg_power_spectrum)


def rotate90(image):
    """Rotate an image by 90 degrees.
    
    parameters:
    - image: a numpy.ndarray"""
    
    grey_channel = rgb2grey(image)
    new_channel = np.transpose(grey_channel, axes=(1,0))[::-1,:]
    return np.dstack((new_channel,new_channel,new_channel))

def rotate180(image):
    """Rotate an image by 180 degrees.
    
    parameters:
    - image: a numpy.ndarray"""
    
    grey_channel = rgb2grey(image)
    new_channel = grey_channel[::-1,::-1]
    return np.dstack((new_channel,new_channel,new_channel))

def rotate270(image):
    """Rotate an image by 270 degrees.
    
    parameters:
    - image: a numpy.ndarray"""
    
    grey_channel = rgb2grey(image)
    new_channel = np.transpose(grey_channel[::-1,:], axes=(1,0))
    return np.dstack((new_channel,new_channel,new_channel))

###########################################################
#   HELPER FUNCTIONS
###########################################################

def apply_uniform_noise(image, low, high, rng=None):
    """Apply uniform noise to an image, clip outside values to 0 and 1.

    parameters:
    - image: a numpy.ndarray 
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    nrow = image.shape[0]
    ncol = image.shape[1]

    image = image + get_uniform_noise(low, high, nrow, ncol, rng)

    #clip values
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)

    assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"

    return image


def get_uniform_noise(low, high, nrow, ncol, rng=None):
    """Return uniform noise within [low, high) of size (nrow, ncol).

    parameters:
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - nrow: number of rows of desired noise
    - ncol: number of columns of desired noise
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    if rng is None:
        return np.random.uniform(low=low, high=high,
                                 size=(nrow, ncol))
    else:
        return rng.uniform(low=low, high=high,
                           size=(nrow, ncol))


def is_in_bounds(mat, low, high):
    """Return wether all values in 'mat' fall between low and high.

    parameters:
    - mat: a numpy.ndarray 
    - low: lower bound (inclusive)
    - high: upper bound (inclusive)
    """

    return np.all(np.logical_and(mat >= 0, mat <= 1))


def eidolon_partially_coherent_disarray(image, reach, coherence, grain):
    """Return parametrically distorted images (produced by Eidolon factory.

    For more information on the effect of different distortions, please
    have a look at the paper: Koenderink et al., JoV 2017,
    Eidolons: Novel stimuli for vision research).

    - image: a numpy.ndarray
    - reach: float, controlling the strength of the manipulation
    - coherence: a float within [0, 1] with 1 = full coherence
    - grain: float, controlling how fine-grained the distortion is
    """

    # import here to make other code not dependent on eidolon package
    import wrapper as wr
    return wr.partially_coherent_disarray(wr.data_to_pic(image),
                                          reach, coherence, grain)


def get_opponent_colours(image):
    """Return image converted to have opponent colours.
    
    parameters:
    - image: a numpy.ndarray"""

    # define matrix describing the cone spectral sensitivities
    # for later conversion from CMFs to respective cone activations
    # [l_r l_g l_b] as sensitivity of the long_wave receptor
    # values calculated from monitor spectrum for [FF0000], [00FF00], [0000FF] and
    # 2-deg LMS fundamentals based on the Stiles and Burch 10-deg CMFs adjusted to 2-deg
    M = np.array([[0.1619982647, 0.208049264, 0.0389347448],
                  [0.0330275785, 0.2496504557, 0.0618343518],
                  [0.0015524055, 0.0198861001, 0.1890224274]])

    # calculate individual differential cone activities from red, green, blue channels
    # use white_point as background
    cones = np.einsum('ij,klj->kli', M, image)
    #white_point = np.array([0.5, 0.5, 0.5])
    cones_white_point = [ 0.20449114,  0.17225619,  0.10523047]
    delta_cones = cones - cones_white_point

    # define matrix describing conversion from cone activities to colour opponent channels
    # following Brainard, D. H. (1996). Cone contrast and opponent modulation color spaces.
    N = np.array([[9.27590805, 9.27590805, 0],
                  [7.86421647, -9.33587662, 0],
                  [-3.39709047, -3.39709047, 12.16230246]])

    # calculate opponent colour channels from cone activations
    dkl = np.einsum('ij,klj->kli', N, delta_cones)
    L_M = dkl[:,:,1]
    S_Lum = dkl[:,:,2]

    # apply transformation on opponent colour channels
    # transform image to have opponent colours
    dkl[:,:,1] = -1 * L_M
    dkl[:,:,2] = -1 * S_Lum

    # recalculate cone activations from opponent colour channels
    new_cones = np.einsum('ij,klj->kli', np.linalg.inv(N), dkl)
    new_cones = new_cones + cones_white_point

    # recalculate rgb-channels from cone activities
    return np.einsum('ij,klj->kli', np.linalg.inv(M), new_cones)

def find_nearest(value, x_gamma_function):
    """Find the nearest element in x_gamma_function to value"""
    idx = (np.abs(x_gamma_function-value)).argmin()
    return idx

def find_nearest_vec(tensor, x_gamma_function):
    """Vectorise "find_nearest" function to use with tensor instead of a single value"""
    vec = np.vectorize(lambda t: find_nearest(t, x_gamma_function))
    return vec(tensor)

def monitor_nonlinearity(x, x_gamma_function):
    """Monitor colour displaying is not linear but follows x^gamma function.
    Convert raw image values to monitor scale.
    
    parameters:
    - x: image as numpy.ndarray (floats in [0,255])"""

    x = x * 255
    x_int = x.astype(np.uint8)
    assert np.allclose(x, x_int)

    return x_gamma_function[x_int].reshape(x.shape)

def inv_monitor_nonlinearity(x, x_gamma_function):
    """Monitor colour displaying is not linear but follows x^gamma function.
    Convert monitor scale back to linear image float values.
    
    parameters:
    - x: image as numpy.ndarray (floats in [0,1])"""

    # clip values that are too small or too large
    x[x > 1] = 1
    x[x < 0] = 0

    # find the value in the x_gamma_function that closest corresponds to
    # each value in the image. 
    y = find_nearest_vec(x, x_gamma_function)
    return y / 255

def scramble_phases(image, width):
    """Apply random shifts to an images' frequencies' phases in the Fourier domain.
    
    parameter:
    - image: an numpy.ndaray
    - width: maximal width of the random phase shifts"""

    # create array with random phase shifts from the interval [-width,width]
    length = (image.shape[0]-1)*(image.shape[1]-1)
    phase_shifts = np.random.random(length//2) - 0.5
    phase_shifts = phase_shifts * 2 * width/180 * np.pi
    
    # convert to graysclae
    channel = rgb2grey(image)

    # Fourier Forward Tranform and shift to centre
    f = fp.fft2(channel) #rfft for real values
    f = fp.fftshift(f)

    # get amplitudes and phases
    f_amp = np.abs(f)
    f_phase = np.angle(f)

    # transformations of phases
    # just change the symmetric parts of FFT outcome, which is
    # [1:,1:] for the case of even image sizes
    fnew_phase = f_phase
    fnew_phase[1:,1:] = shift_phases(f_phase[1:,1:], phase_shifts)

    # recalculating FFT complex representation from new phases and amplitudes
    fnew = f_amp*np.exp(1j*fnew_phase)

    # reverse shift to centre and perform Fourier Backwards Transformation
    fnew = fp.ifftshift(fnew)
    new_channel = fp.ifft2(fnew)

    # make sure that there are no imaginary parts after transformation
    new_channel = new_channel.real

    # clip too small and too large values
    new_channel[new_channel > 1] = 1
    new_channel[new_channel < 0] = 0

    # return stacked (RGB) grey image
    return np.dstack((new_channel, new_channel, new_channel))

def shift_phases(f_phase, phase_shifts):
    """Applies phase shifts to an array of phases.
    
    parameters: 
    - f_phase: the original images phases (in frequency domain)
    - phase_shifts: an array of phase shifts to apply to the original phases 
    """

    # flatten array for easier transformation
    f_shape = f_phase.shape
    flat_phase = f_phase.flatten()
    length = flat_phase.shape[0]

    # apply phase shifts symmetrically to complex conjugate frequency pairs
    # do not change c-component
    flat_phase[:length//2] += phase_shifts
    flat_phase[length//2+1:] -= phase_shifts

    # reshape into output format
    f_phase = flat_phase.reshape(f_shape)

    return f_phase


def equalise_power_spectrum(image, avg_power_spectrum):
    """Equalise images' power spectrum by setting an image's amplitudes 
    in the Fourier domain to the amplitude average over all used images.
    
    parameter:
    - image: a numpy.ndarray
    - avg_power_spectrum: an array of the same dimension as one of images channels
                          containing the average over all images amplitude spectrum"""
    
    # check input dimensions
    assert image.shape[:2] == avg_power_spectrum.shape, 'Image shape={} unequal \
            avg_spectrum shape={}'.format(image.shape[:2], avg_power_spectrum.shape)

    # convert image to greysclae
    channel = rgb2grey(image)

    # Fourier Forward Tranform and shift to centre
    f = fp.fft2(channel)
    f = fp.fftshift(f)

    # get amplitudes and phases
    f_amp = np.abs(f)
    f_phase = np.angle(f)

    # set amplitudes to average power spectrum
    fnew_amp = avg_power_spectrum

    # recalculating FFT complex representation from new phases and amplitudes
    fnew = fnew_amp*np.exp(1j*f_phase)

    # reverse shift to centre and perform Fourier Backwards Transformation
    fnew = fp.ifftshift(fnew)
    new_channel = fp.ifft2(fnew)

    # make sure that there are no imaginary parts after transformation
    new_channel = new_channel.real

    # clip too large and too small values
    new_channel[new_channel > 1] = 1
    new_channel[new_channel < 0] = 0

    # return stacked (RGB) grey image
    return(np.dstack((new_channel, new_channel, new_channel)))

###########################################################
#   MAIN METHOD FOR TESTING & DEMONSTRATION PURPOSES
###########################################################

if __name__ == "__main__":

    print("""This main method should generate manipulated
           images in the directory where it was executed.""")

    use_JPEG = False # either JPEG or PNG
    img = imload_rgb("test_image.JPEG")

    ###################################################
    # A) Example for color-experiment:
    #    - convert to grayscale
    ###################################################

    img_grayscale = rgb2gray(img)
    save_img(img_grayscale, "test_image_grayscale", use_JPEG)

    ###################################################
    # B) Example for contrast-experiment:
    #    - convert to grayscale and
    #    - reduce contrast to nominal contrast of 10%
    ###################################################

    contrast_level_1 = 0.1 
    img_low_contrast = grayscale_contrast(image=img,
                                          contrast_level=contrast_level_1)
    save_img(img_low_contrast, "test_image_low_contrast", use_JPEG)

    ###################################################
    # C) Example for noise-experiment:
    #    - convert to graycale and
    #    - reduce contrast to 30% and
    #    - apply uniform noise with width 0.1
    ###################################################

    noise_width = 0.1
    contrast_level_2 = 0.3
    rng = np.random.RandomState(seed=42)

    img_noisy = uniform_noise(image=img, width=noise_width,
                              contrast_level=contrast_level_2,
                              rng=rng)
    save_img(img_noisy, "test_image_noisy", use_JPEG)

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
    save_img(img_noisy, "test_image_salt_and_pepper_noise", use_JPEG)

    ###################################################
    # E) Example for eidolon-experiment:
    #    - use partially_coherent_disarray 
    ###################################################

    grain = 10.0
    coherence = 1.0
    reach = 8.0

    img_eidolon = eidolon_partially_coherent_disarray(img, reach,
                                                      coherence, grain) 
    save_img(img_eidolon, "test_image_eidolon", use_JPEG)
    
    ###################################################
    # F) Example for false-colour-experiment: 
    ###################################################

    # load function for monitor non-linearity
    x_gamma_function = np.load('./x_gamma_function.npy')

    img_false_colour = false_colour(img, x_gamma_function) 
    save_img(img_false_colour, "test_image_false-colour", use_JPEG)
    
    ###################################################
    # G) Example for highpass-experiment: 
    #    - use a standard devation of 3
    ###################################################
    
    std = 3

    img_highpass = high_pass_filter(img, std) 
    save_img(img_highpass, "test_image_highpass", use_JPEG)
    
    ###################################################
    # H) Example for lowpass-experiment: 
    #    - use a standard devation of 10
    ###################################################
    
    std =10

    img_lowpass = low_pass_filter(img, std) 
    save_img(img_lowpass, "test_image_lowpass", use_JPEG)
    
    ###################################################
    # I) Example for phase-scrambling: 
    #    - use a noise width of 90 degrees
    ###################################################
    
    width = 90

    img_phase_scrambling = phase_scrambling(img, width) 
    save_img(img_phase_scrambling, "test_image_phase-scrambling", use_JPEG)
    
    ###################################################
    # J) Example for power equalisation: 
    ###################################################

    # load mean amplitude spectrum over all images
    avg_power_spectrum = np.load('./mean_power_spectrum_grey.npy')

    img_power_equalisation = power_equalisation(img, avg_power_spectrum) 
    save_img(img_power_equalisation, "test_image_power_equalisation", use_JPEG)
    
    ###################################################
    # K) Example for rotation: 
    ###################################################

    img_rotation90 = rotate90(img) 
    save_img(img_rotation90, "test_image_rotation90", use_JPEG)
    
    img_rotation180 = rotate180(img) 
    save_img(img_rotation180, "test_image_rotation180", use_JPEG)
    
    img_rotation270 = rotate270(img) 
    save_img(img_rotation270, "test_image_rotation270", use_JPEG)
