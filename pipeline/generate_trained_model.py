'''
This generates one trained model, predicts on some test data, and marks the false positives/false negatives.
'''

import numpy as np
import glob
import cPickle as pickle
import time
import image_processing
from sklearn.linear_model import LogisticRegression
from collections import Counter

def confusion_matrix(predicted, actual):
    '''
    | Outputs (what model thinks it is, what it is)

    '''
    predicted, actual = map(np.array, [predicted, actual])
    return Counter(zip(predicted, actual))

def _get_params(median=False, median_smooth__radius=3, hog__orientations=9, 
                hog__pixels_per_cell=(8,8), hog__cells_per_block=(1,1), 
                logistic_regression__C=0.1) :
    return locals()


def succesful_predictions(kw) :
    '''
    | Create a pickle file of the trained model
    | returns successful predictions of the test data
    '''
    
    # Turn the tuple into a packed dictionary to get all parameters
    params = _get_params(logistic_regression__C=kw) 

    # Create the pipeline which consists of image
    # processing and a classifier
    # Note - can make this map to a dictionary of image processors instead of just median
    image_processors = [ ('hog', image_processing.HOG()) ]
    if params.pop('median') :
        image_processors.insert(0,('median_smooth', image_processing.MedianSmooth()))
    else :
        params.pop('median_smooth__radius')

    classifier = ('logistic_regression', LogisticRegression())

    estimators = image_processors + [classifier]
    
    pipeline = Pipeline(estimators)

    # Create the grid search with list of parameters
    # to search.  All values are now tuples
    pipeline.set_params(**params).fit(X_train, y_train)

    with open('LogReg10.pkl','wb') as output :
        pickle.dump(pipeline, output, -1)

    # Confusion matrix is                                                                                                                            
    print "Confusion matrix on training set"
    print confusion_matrix(pipeline.predict(X_train), y_train)
    print
    print "Score on training set =", pipeline.score(X_train, y_train)
    print

    # Score the test set                                                                                                                              
    print "Confusion matrix on test set"
    print confusion_matrix(pipeline.predict(X_test), y_test)
    print
    print "Score on test set =", pipepine.score(X_test, y_test)

    return map(lambda x: x[0] == x[1], zip(pipeline.predict(X_test), y_train) )


def get_false_predictions_list( successful_predictions, filenames ) :

    
    return [ sf[1] for sf in zip( successful_predictions, filenames ) if sf[0] == 0 ]

def generate_X_y(non_lens_glob, lens_glob) :
    '''Reads in data that will be features and targets, outputs as numpy array data'''
    non_lens_filenames = glob.glob(non_lens_glob)
    lens_filenames = glob.glob(lens_glob)
    filenames = non_lens_filenames + lens_filenames
    X = image_processing.load_images(filenames)
    y = [0] * len(non_lens_filenames) + [1] * len(lens_filenames)
    
    return X, y, filenames
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('non_lens_glob')
    parser.add_argument('lens_glob')
    parser.add_argument('non_lens_test_glob')
    parser.add_argument('lens_test_glob')

    args = vars(parser.parse_args())

    C_val = 10.
    rotation_degrees = [ 0, 90, 180, 270 ]
    parameters = [('logistic_regression__C',C_val) for C_val in C_vals ]
    
    # Load the data. X is a list of numpy arrays
    # which are the images.
    X_train, y_train, filenames_train = generate_X_y( args['non_lens_glob'], args['lens_glob'] )
    X_test, y_test, filenames_test = generate_X_y( args['non_lens_test_glob'], args['lens_test_glob'] )


    # Train/test split
    X_train, y_train = image_processing.rotate_images( rotation_degrees, X_train, y_train )
    
    print "False Predictions: ", get_false_predictions_list( successful_predictions(C_val), filenames_test )

    start_time = time.time()
    time_taken = time.time() - start_time
    print "Time to generate object:", time_taken

