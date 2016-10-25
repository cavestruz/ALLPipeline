import numpy as np
from collections import Counter

def confusion_matrix(predicted, actual):
    '''
    | Outputs (what model thinks it is, what it is)

    '''
    predicted, actual = map(np.array, [predicted, actual])
    return Counter(zip(predicted, actual))

def print_model_scores(model, X_train, y_train, X_test, y_test) :
    ''' 
    | Outputs scores from training and testing sets
    | The input model arg must have predict and score as methods
    '''


    # Show score for training set using best parameters
    # Confusion matrix is 
    print "Confusion matrix on training set"
    print confusion_matrix(model.predict(X_train), y_train)
    print
    print "Score on training set =", model.score(X_train, y_train)
    print

    # Score the test set
    print "Confusion matrix on test set"
    print confusion_matrix(model.predict(X_test), y_test)
    print
    print "Score on test set =", model.score(X_test, y_test)

def get_false_predictions_list( trained_model, X, y, filenames ) :
    '''
    |    Trained model must have predict and score methods.
    |    X, y and filenames must have same length
    '''
    assert( len(X) == len(y) ) 
    assert( len(y) == len(filenames) )

    # Score the test set                                                                
    print "Confusion matrix on test set"
    print confusion_matrix(trained_model.predict(X), y)
    print
    print "Score on test set =", trained_model.score(X, y)


    successful_predictions = map( lambda x: x[0] == x[1], 
                                  zip( trained_model.predict( X ), y) )
    
    return [ sf[1] for sf in zip( successful_predictions, filenames ) if sf[0] == 0 ]

