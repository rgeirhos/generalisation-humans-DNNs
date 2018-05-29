#!/usr/bin/env python
"""wrapper.py

A small set of functions used to make the eidolon factory code a little easier
to integrate and work with.

The 'eidolon factory' can be obtained from here:
https://github.com/gestaltrevision/Eidolon

Note that the Python version likely contains a bug, for which we opened
a pull request with a bugfix on 3rd August 2016 (#1). The bugfix
has not (yet?) been merged as of May 2017, so you'll need to manually
replace the files from the pull request.

The corresponding paper ("Eidolons: Novel stimuli for vision research")
can be obtained here:
http://jov.arvojournals.org/article.aspx?articleid=2603227

"""
from __future__ import division

from PIL import Image
from tempfile import mkstemp
import numpy as np
import os
from skimage.io import imsave

# you'll need to include the eidolon/ directory of the Eidolon
# toolbox for these imports to work. See above for information on
# how to obtain this.
from eidolon import picture as pic
from eidolon import helpers as hel
from eidolon import scalespaces as scl
from eidolon import noise as noi


SZ = 256
MIN_STD = (1 / np.sqrt(2))
MAX_STD = SZ / 4.0
STD_FAC = np.sqrt(2)


def load_pic(fname, sz=SZ, min_s=MIN_STD, max_s=MAX_STD, s_factor=STD_FAC):
    """Just make a picture using sane defaults"""
    return pic.Picture(fname, sz, min_s, max_s, s_factor)


def data_to_pic(dat, sz=SZ, min_s=MIN_STD, max_s=MAX_STD, s_factor=STD_FAC):
    """Turn a matrix into an Eidolon Picture object"""

    (outfd, fname) = mkstemp('.png')

    imsave(fname, dat)

    # close temp file, otherwise one gets complaints about
    # too many open files
    outsock = os.fdopen(outfd, 'w')
    outsock.close()

    return load_pic(fname, sz, min_s, max_s, s_factor)


def pic_to_data(pic):
    """Return the data inside an Eidolon Picture object"""
    return np.asarray(pic.resizedOriginalImage)


def coherent_disarray(pic, reach):
    """Return a coherently scattered image"""

    (h, w) = pic.fatFiducialDataPlane.shape

    dog_scale_space = scl.DOGScaleSpace(pic)
    ei_dat = hel.CoherentDisarray(dog_scale_space, reach, w, h,
                                  pic.MAX_SIGMA, pic.numScaleLevels,
                                  pic.scaleLevels)

    eidolon = pic.DisembedDataPlane(ei_dat)
    return np.asarray(Image.fromarray(eidolon, 'L'))


def superficial_disarray(pic, reach,  grain):

    (h, w) = pic.fatFiducialDataPlane.shape
    b1 = noi.BlurredRandomGaussianDataPlane(w, h, grain)
    xDisplacements = b1.blurredRandomGaussianDataPlane
    b2 = noi.BlurredRandomGaussianDataPlane(w, h, grain)
    yDisplacements = b2.blurredRandomGaussianDataPlane
    eidolonDataPlane = hel.DataPlaneDisarray(pic.fatFiducialDataPlane,
                                             xDisplacements, yDisplacements,
                                             reach)

    eidolon = pic.DisembedDataPlane(eidolonDataPlane)
    return np.asarray(Image.fromarray(eidolon, 'L'))


def lotze_disarray(pic, reach, grain):
    (h, w) = pic.fatFiducialDataPlane.shape
    fiducialDOGScaleSpace = scl.DOGScaleSpace(pic)

    ei_dat = hel.LotzeDisarray(fiducialDOGScaleSpace, reach, grain,
                               pic.numScaleLevels, w, h)

    eidolon = pic.DisembedDataPlane(ei_dat)
    return np.asarray(Image.fromarray(eidolon, 'L'))


def helmholtz_disarray(pic, reach):
    (h, w) = pic.fatFiducialDataPlane.shape

    fiducialDOGScaleSpace = scl.DOGScaleSpace(pic)

    eidolonDataPlane = hel.HelmholtzDisarray(fiducialDOGScaleSpace, reach,
                                             pic.numScaleLevels, w, h,
                                             pic.MAX_SIGMA, pic.scaleLevels)

    eidolon = pic.DisembedDataPlane(eidolonDataPlane)
    return np.asarray(Image.fromarray(eidolon, 'L'))


def coherent_disarray_of_edges(pic, reach):

    (h, w) = pic.fatFiducialDataPlane.shape
    fiducialDOGScaleSpace = scl.DOGScaleSpace(pic)

    eidolonDataPlane = hel.CoherentDisarray(fiducialDOGScaleSpace, reach, w, h,
                                            pic.MAX_SIGMA, pic.numScaleLevels,
                                            pic.scaleLevels)

    eidolon = pic.DisembedDataPlane(eidolonDataPlane)
    return np.asarray(Image.fromarray(eidolon, 'L'))


def partially_coherent_disarray(pic, reach, coherence, grain):

    fiducialDOGScaleSpace = scl.DOGScaleSpace(pic)
    (h, w) = pic.fatFiducialDataPlane.shape

    eidolonDataPlane = hel.PartiallyCoherentDisarray(fiducialDOGScaleSpace,
                                                     reach, coherence, grain,
                                                     w, h, pic.MAX_SIGMA,
                                                     pic.numScaleLevels,
                                                     pic.scaleLevels)

    eidolon = pic.DisembedDataPlane(eidolonDataPlane)
    return np.asarray(Image.fromarray(eidolon, 'L'))
