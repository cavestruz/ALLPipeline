import numpy as np
import copy

def load_images(filenames):
    '''Expects filenames to be a list of .fits file locations'''
    from astropy.io.fits import getdata

    return [ normalized_positive_image(getdata(filename)) for filename in filenames ]

def normalized_positive_image(image) :
    '''Images need to be numpy arrays between -1 and 1 for median and
    possibly HOG, but also should be log normalized so contrast is
    maintained.'''
    
    pos_def = np.clip(image,1e-6,1e100)+1.0
    return np.log(pos_def) / abs(np.log(pos_def)).max()

