import numpy as np
import copy
from sklearn.base import BaseEstimator

def load_images(filenames):
    '''Expects filenames to be a list of .fits file locations'''
    from astropy.io.fits import getdata
    return [getdata(filename) for filename in filenames]

class MedianSmooth(BaseEstimator):
    def __init__(self, radius = 3):
        self.radius = radius

    def fit(self, images, y = None):
        return self

    def transform(self, images):
        from skimage.filters.rank import median
        from skimage.morphology import disk
        return np.array([median(image, disk(self.radius))
                         for image in images])

    def fit_transform(self, images, y = None):
        return self.transform(images)

class HOG(BaseEstimator):
    def __init__(self, orientations = 9, pixels_per_cell = (8, 8),
                 cells_per_block = (3, 3), normalise = False):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.normalise = normalise

    def fit(self, images, y = None):
        return self

    def transform(self, images):
        from skimage.feature import hog
        return np.array([hog(image,
                             orientations = self.orientations,
                             pixels_per_cell = self.pixels_per_cell,
                             cells_per_block = self.cells_per_block,
                             normalise = self.normalise)
                         for image in images])

    def fit_transform(self, images, y = None):
        return self.transform(images)
