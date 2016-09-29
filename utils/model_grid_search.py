import numpy as np
import sys
import glob
import time
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import image_processing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter

def parse_configfile(cfgfile) :
    '''Return a nested dictionary for parameter grid and pipeline'''
    import ConfigParser
    cfg_dict = {}
    Config = ConfigParser.ConfigParser()
    Config.read(cfgfile)
    for section in Config.sections() : 
        cfg_dict[section] = { option: Config.get(section, option) \
                                  for option in Config.options(section) }

    return cfg_dict
        

def build_parameter_grid(param_grid) :
    return [ param_grid ]

def build_pipeline(image_processors, classifier) :
    # Create the pipeline which consists of image
    # processing and a classifier
    image_processors = [('hog', image_processing.HOG())]
    classifier = ('logistic_regression', LogisticRegression())
    estimators = image_processors + [classifier]

    return Pipeline(estimators)

def grid_search() :
    pass

if __name__ == "__main__" :
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('non_lens_glob')
    parser.add_argument('lens_glob')

    args = vars(parser.parse_args())
    rotation_degrees = [ 0, 90, 180, 270 ]
    
    # Load the data. X is a list of numpy arrays
    # which are the images.
    non_lens_filenames = glob.glob(args['non_lens_glob'])
    lens_filenames = glob.glob(args['lens_glob'])
    filenames = non_lens_filenames + lens_filenames
    X = image_processing.load_images(filenames)
    y = [0] * len(non_lens_filenames) + [1] * len(lens_filenames)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2)
    X_train, y_train = image_processing.rotate_images(rotation_degrees,X_train, y_train)

    print "len(X_train) =", len(X_train)
    print "len(y_train) =", len(y_train)
    print "len(X_test) =", len(X_test)
    print "len(y_test) =", len(y_test)
    print


    pipeline = Pipeline(estimators)

    # Create the grid search with list of parameters
    # to search
    param_grid = [{'hog__orientations' : (9,),#, 10),
                   'hog__pixels_per_cell' : ((8, 8),(4, 4),(2, 2)),#, (16, 16)),
                   'hog__cells_per_block' : ((1, 1),(2, 2),(3, 3)),
                   'logistic_regression__C' : (1., 5., 10., 50.,) #Regularization parameter
                   },
                  ]

    grid_search = GridSearchCV(pipeline, param_grid,
                               n_jobs = -1)

    # Train the model on the training set
    print "Running grid search..."
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    time_taken = time.time() - start_time
    print "Finished grid search. Took", time_taken, "seconds"
    print
    print "Best score:", grid_search.best_score_
    print "Best parameters set:",
    best_parameters = grid_search.best_estimator_.get_params()
    param_names = reduce(lambda x, y : x | y,
                         (p_grid.viewkeys()
                          for p_grid in param_grid))
    for param_name in sorted(param_names):
        print param_name, ":", best_parameters[param_name]
    print

    print "Scores for each parameter combination:"

    #  Splits into a subcycle of something like groups of 3 subsets to
    #  test out the parameters in the grid and select a final best
    #  mean.
    #  http://scikit-learn.org/stable/modules/cross_validation.html
    for grid_score in grid_search.grid_scores_:
        print grid_score
    print

    # Show score for training set using best parameters
    # Confusion matrix is 
    print "Confusion matrix on training set"
    print confusion_matrix(grid_search.predict(X_train), y_train)
    print
    print "Score on training set =", grid_search.score(X_train, y_train)
    print

    # Score the test set
    print "Confusion matrix on test set"
    print confusion_matrix(grid_search.predict(X_test), y_test)
    print
    print "Score on test set =", grid_search.score(X_test, y_test)
