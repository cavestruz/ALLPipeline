import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, Imputer, Normalizer

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

class Norm(BaseEstimator) : 
    def __init__( self, axis=None ) :
        self.axis = axis

    def fit( self, images, y = None ) :
        return self

    def transform( self, images ) :
        from numpy.linalg import norm
        return np.array( [ image/norm(image) for image in images ] ) 
    
    def fit_transform( self, images, y = None ) :
        return self.transform( images )

        
class Clip(BaseEstimator) :
    '''Numpy clip'''

    def __init__( self, lower=1e-6, upper=1e100):

        self.clip_min = lower
        self.clip_max = upper

    def fit(self, image, y = None) : 

        return self

    def transform( self, images ) :

        return np.array( [ np.clip( image, self.clip_min, self.clip_max ) 
                           for image in images ] )

    def fit_transform( self, images, y = None ) :

        return self.transform(images)

class LogPositiveDefinite(BaseEstimator) :
    ''' Shift all values to positive definite with options for taking
    log of image and making log positive definite. Return normalized
    values.'''

    def __init__( self, log = True, shift = None ) : 
        self.log = log
        self.shift = shift

    def fit( self, images, y = None ) :
        return self

    def _make_positive( self, image ) :
        ''' Ensure that the minimum value is just above zero. '''
        if self.shift == None :
            return image - image.min() + np.abs(image.min())
        else :
            return image + self.shift

    def _normalize( self, image ) :
        if self.log : 
            return np.log( self._make_positive(image) ) / np.log( self._make_positive(image) ).max()
        else :
            return self._make_positive(image) / self._make_positive(image).max()

    def transform( self, images ) :
        return np.array( [ self._normalize(image) for image in images ] )

class PreprocessHST( BaseEstimator ) :
    '''This is significantly the best set of preprocessing I've found
    for our HST like data.'''
    def __init__( self, lower_clip = 1e-6, upper_clip = 1e100, shift = 1.0 ):
        self.lower_clip = lower_clip
        self.upper_clip = upper_clip
        self.shift = shift

    def fit( self, images, y=None ) :
        return self

    def _pos_def( self, image ) :
        return np.clip(image, self.lower_clip, self.upper_clip) + self.shift

    def transform( self, images ) :
        return np.array( [ np.log(self._pos_def(image)) / abs(np.log(self._pos_def(image))).max() for image in images ] )

    def fit_transform( self, images, y = None ) :
        return self.transform( images ) 

class MidpointSigmaClip(BaseEstimator) :
    '''Clips the data at n sigma from a midpoint.
    kwargs
    ---------
    sigma_factor : -1 means 1 sigma below the midpoint
    mid_point : option for calculating midpoint, e.g. 'median', 'mean'
    '''
    def __init__( self, sigma_factor=-1., mid_point='mean', upper_clip=1e100, normalize=0. ):
        self.sigma_factor = sigma_factor
        self.upper_clip = upper_clip
        if normalize == 1. : self.normalize = True
        else : self.normalize = False
        mid_point_options = ['median', 'mean' ]
                                   
        assert( mid_point in mid_point_options )
        self.mid_point = mid_point

    def fit( self, images, y = None ) :
        return self
    
    def _calc_lower_clip( self, image ) :
        if self.mid_point == 'median' :
            return np.median( image ) + image.std() * self.sigma_factor
        if self.mid_point == 'mean' :
            return np.mean( image ) + image.std() * self.sigma_factor

    def _clip( self, image ) :
        return np.clip(image, self._calc_lower_clip(image), self.upper_clip)

    def transform( self, images ) :
        if self.normalize : 
            from numpy.linalg import norm
            #from sklearn.preprocessing import normalize
            return np.array([ self._clip(image)/norm(self._clip(image)) for image in images])
        else :
            return np.array([ self._clip(image) for image in images])

    def fit_transform( self, images, y = None ) :
        return self.transform( images )

class HOG(BaseEstimator):
    def __init__( self, orientations = 9, pixels_per_cell = (8, 8),
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
                     'clip' : Clip(),
                     'log_positive_definite' : LogPositiveDefinite(),
                     'scale' : StandardScaler(),
                     'imputer' : Imputer(),
                     'hst' : PreprocessHST(),
                     'midpointsigmaclip' : MidpointSigmaClip(),
                     'norm' : Norm(),
                     'normalizer' : Normalizer(),
                     }

