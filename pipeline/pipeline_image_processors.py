import numpy as np
from sklearn.base import BaseEstimator

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
                 cells_per_block = (3, 3) ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, images, y = None):
        return self

    def transform(self, images):
        from skimage.feature import hog
        return np.array([hog(image,
                             orientations = self.orientations,
                             pixels_per_cell = self.pixels_per_cell,
                             cells_per_block = self.cells_per_block,
                             )
                         for image in images])

    def fit_transform(self, images, y = None):
        return self.transform(images)


image_processors = { 'median_filter' : MedianSmooth(),
                     'hog' : HOG(),    
                     }
