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

