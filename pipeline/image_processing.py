import numpy as np
import copy

def load_images(filenames):
    from skimage import io
    return [io.imread(filename) for filename in filenames]

class MedianSmooth:
    def __init__(self, radius):
        self._radius = radius

    def fit(self, images, y = None):
        return self

    def transform(self, images):
        from skimage.filters.rank import median
        from skimage.morphology import disk
        return np.array([median(image, disk(self._radius))
                         for image in images])

    def fit_transform(self, images, y = None):
        return self.transform(images)

class HOG:
    def __init__(self, **kwargs):
        self._hog_kwargs = copy.deepcopy(kwargs)
        if 'visualise' in self._hog_kwargs:
            del self._hog_kwargs['visualise']

    def fit(self, images, y = None):
        return self

    def transform(self, images):
        from skimage.feature import hog
        return np.array([hog(image, **self._hog_kwargs)
                         for image in images])

    def fit_transform(self, images, y = None):
        return self.transform(images)
