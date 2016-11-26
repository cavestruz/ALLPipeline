import numpy as np
from collections import Counter
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

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

def get_ranked_predictions(trained_model, X, y, filenames):
    '''
    | Returns a list of tuples giving the scores for each
    | filename that is sorted descending by score.
    | trained_model must have predict_proba or
    | decision_function method.
    |
    | Output format:
    |
    | [(filename1, score1, label1),
    |  (filename2, score2, label2),
    |  ...]
    |
    | where filename? is the filename corresponding to
    | a given row of X from the filenames list, score?
    | is the score for the given row of X from
    | trained_model, and label? is the 1/0 label for
    | the given row of X taken from the array y. The
    | output will be sorted such that
    | score1 >= score2 >= score3 >= ...
    '''
    assert( len(X) == len(y) )
    assert( len(y) == len(filenames) )
    
    scores = get_scores(trained_model, X)
    return sorted(zip(filenames, scores, y),
                  key = lambda (filename, score, label) : score,
                  reverse = True)

def calc_tpr_fpr( scores, labels ) :
    '''
    |    Runs down the ordered scores and returns tpr, fpr
    |    corresponding to varied thresholds
    |
    |    Note: the built in roc_curve from sklearn.metrics only
    |    returns the necessary tpr, fpr values to make the plot, and
    |    not the intermediate points for constant tpr or fpr.
    '''
    assert( len(scores) == len(labels) and len(filenames) == len(scores) )
    assert( all( scores[i] >= scores[i+1] for i in xrange(len(scores)-1) )  )

    tpr, fpr = [], []
    num_label1 = float(collections.Counter(labels)[1]))
    num_label0 = float(collections.Counter(labels)[0]))

    for i in len(labels) :
        tpr.append(collections.Counter(labels[:i])[1] / num_label1
        fpr.append(collections.Counter(labels[:i])[0] / num_label0

    return tpr, fpr

def get_tpr_index( tpr, tpr_min, tpr_max, fpr, fpr_min, fpr_max ) :
    '''
    |    Return the indices corresponding to tpr min and max, fpr_min and fpr_max
    '''

    assert( tpr_min >= tpr[-1] and tpr_max <= tpr[0] ) 
    assert( fpr_min >= fpr[-1] and fpr_max <= fpr[0] ) 
    assert( tpr_min <= tpr_max and fpr_min <= fpr_max ) 

    tpr_indices = np.where( (tpr <= tpr_max) & (tpr >= tpr_min) ) 
    fpr_indices = np.where( (fpr <= fpr_max) & (fpr >= fpr_min) ) 

    return tpr_indices, fpr_indices

def get_filenames_in_threshold_range(filenames, labels, tpr_indices, fpr_indices) :
    '''
    |    Return the filenames and corresponding labels that satisfy
    |     the tpr or fpr range by broadcasting.
    '''
    
    return zip( filenames[tpr_indices], labels[tpr_indices] ), \
                       zip( filenames[fpr_indices], labels[fpr_indices] )


def roc_curve_data(model, X, y):
    '''
    | Outputs the ROC curve for the given model and data.
    | model must have predict_proba or decision_function
    | method.
    |
    | The output is two arrays:
    | 1. fpr = False positive rate
    | 2. tpr = True positive rate
    |
    | A ROC plot shows fpr on the x axis and tpr on the
    | y axis.  More info:
    | http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    '''
    scores = get_scores(model, X)
    fpr, tpr, _ = roc_curve(y, scores)
    return fpr, tpr

def roc_curve_plot(model, X, y, outfile):
    '''
    | Plots the ROC curve for the given model and data.
    | Uses roc_curve_data to get the points for the plot,
    | and then saves a pdf of the plot to outfile. Also
    | returns the data points in the plot in two arrays:
    | fpr, tpr.
    '''
    fpr, tpr = roc_curve_data(model, X, y)
    plt.clf()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle = '--', color = 'k')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(outfile)
    return fpr, tpr

def roc_auc(model, X, y):
    '''
    | Outputs the area under the ROC curve for the given
    | model and data.  model must have predict_proba or
    | decision_function method.
    | The output is the area under the ROC curve (AUC).
    | More info:
    | http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    '''
    scores = get_scores(model, X)
    return roc_auc_score(y, scores)

def model_coeff_plot(model, outfile):
    '''
    | Plots the coefficents of the model and saves to
    | outfile.  model must have model.coef_ attribute.
    | Also returns the array of coefficients that are
    | plotted.
    '''
    data = np.reshape(model.coef_, (model.coef_.shape[1],))
    plt.clf()
    plt.plot(data)
    plt.savefig(outfile)
    return data

def get_scores(model, X):
    '''
    | Returns the scores for each point. model must
    | have predict_proba or decision_function method.
    '''
    index_of_positive_class = np.where(model.classes_ == 1)[0][0]
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:,index_of_positive_class]
    elif hasattr(model, 'decision_function'):
        return model.decision_function(X)
    else:
        raise Exception("model must have method predict_proba or decision_function")
