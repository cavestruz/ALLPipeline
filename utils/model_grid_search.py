import numpy as np
import sys

def parse_configfile(cfgfile) :
    '''
    |
    |   Return a nested dictionary for parameter grid and pipeline
    |
    '''
    import ConfigParser
    cfg_dict = {}
    Config = ConfigParser.ConfigParser()
    Config.optionxform = str # Otherwise, options are lowercased
    Config.read(cfgfile)
    for section in Config.sections() : 
        cfg_dict[section] = { option: Config.get(section, option) \
                                  for option in Config.options(section) }

    return cfg_dict        

def load_train_test_data(non_lens_glob, lens_glob, rotation_degrees=None) :
    '''
    |
    |   Load the data. 
    |   X is a list of numpy arrays which are the images. 
    |
    '''
    import StrongCNN.IO.load_images as load_images 
    import StrongCNN.IO.augment_images as augment_images
    from sklearn.cross_validation import train_test_split
    import glob

    non_lens_filenames = glob.glob(non_lens_glob)
    lens_filenames = glob.glob(lens_glob)
    filenames = non_lens_filenames + lens_filenames
    X = load_images.load_images(filenames)
    y = [0] * len(non_lens_filenames) + [1] * len(lens_filenames)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2)

    # Rotate the training set
    #degrees = [ float(rotation_degree) for rotation_degree in rotation_degrees.values() ]
    #X_train, y_train = augment_images.rotate_images(rotation_degrees,X_train, y_train)

    return X_train, y_train, X_test, y_test

def build_parameter_grid(param_grid) :
    import ast
    return [ {k: ast.literal_eval(v) for k,v in param_grid.iteritems()} ]


def build_pipeline(image_processor_labels, classifier_label) :
    '''
    |
    |   Create the pipeline which consists of 
    |   image processing step(s) and a classifier
    |
    '''

    from sklearn.pipeline import Pipeline

    from StrongCNN.utils.pipeline_image_processors import image_processors
    from StrongCNN.utils.pipeline_classifiers import classifiers

    estimators = []
    for label in image_processor_labels :
        estimators.append((label, image_processors[label]))
    estimators.append((classifier_label, classifiers[classifier_label]))

    return Pipeline(estimators)

def grid_search(pipeline, param_grid) :
    '''
    |
    |   Execute a grid search over the parameter grid for a 
    |   given pipeline, and print results to stdout
    |
    '''
    from sklearn.grid_search import GridSearchCV
    from StrongCNN.utils.model_info import print_model_scores
    import time

    gs = GridSearchCV(pipeline, param_grid, n_jobs = -1)

    # Train the model on the training set
    print "Running grid search..."
    start_time = time.time()
    gs.fit(X_train, y_train)
    time_taken = time.time() - start_time
    print "Finished grid search. Took", time_taken, "seconds"
    print
    print "Best score:", gs.best_score_
    print "Best parameters set:",
    best_parameters = gs.best_estimator_.get_params()
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
    for grid_score in gs.grid_scores_:
        print grid_score
    print

    print_model_scores(gs, X_train, y_train, X_test, y_test)

if __name__ == "__main__":

    cfg = parse_configfile(sys.argv[1])

    X_train, y_train, X_test, y_test = \
        load_train_test_data(cfg['filenames']['non_lens_glob'], 
                             cfg['filenames']['lens_glob'],
                             cfg['rotation_degrees']
                             )
    
    print "len(X_train) =", len(X_train)
    print "len(y_train) =", len(y_train)
    print "len(X_test) =", len(X_test)
    print "len(y_test) =", len(y_test)
    print


    param_grid = build_parameter_grid(cfg['param_grid'])

    pipeline = build_pipeline(cfg['image_processing'].values(), 
                              cfg['classifier']['label'])

    grid_search(pipeline, param_grid) 
