'''
Tools necessary to save a trained model, load a trained model, and to
make predictions from a trained model on a test set and collect false
predictions.
'''

import numpy as np
import glob
import cPickle as pickle
import time
import image_processing
from StrongCNN.utils.model_info import get_false_predictions_list


def _get_params(median=False, median_smooth__radius=3, hog__orientations=9, 
                hog__pixels_per_cell=(16,16), hog__cells_per_block=(1,1), 
                logistic_regression__C=10.) :
    return locals()


def train_model(model_pipeline, X_train, y_train, **params) :
    '''
    | Train the model on given training set and parameters
    '''
    
    # Turn the tuple into a packed dictionary to get all parameters
    params = _get_params(logistic_regression__C=kw) 

    # Create the grid search with list of parameters
    # to search.  All values are now tuples
    model_pipeline.set_params(**params).fit(X, y)


def dump_model(pklfile, trained_model) :
    '''Save a trained model'''

    with open(pklfile,'wb') as output :
        pickle.dump(trained_model, output, -1)
    

def load_model(pklfile) :
    '''Returns a trained model that can make predictions'''

    with open(pklfile,'r') as output :
        return pickle.load(output)


def generate_X_y(non_lens_glob, lens_glob) :
    '''Reads in data that will be features and targets, outputs as numpy array data'''
    non_lens_filenames = glob.glob(non_lens_glob)
    lens_filenames = glob.glob(lens_glob)
    filenames = non_lens_filenames + lens_filenames
    X = image_processing.load_images(filenames)
    y = [0] * len(non_lens_filenames) + [1] * len(lens_filenames)
    
    return X, y, filenames
    
