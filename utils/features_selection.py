import numpy as np
from collections import Counter
from model_info import get_scores

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

def get_ranked_predictions( trained_model, X, y, filenames ):
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

def _calc_tpr_fpr( scores, labels ) :
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

def _get_tpr_fpr_indices( tpr, tpr_min, tpr_max, fpr, fpr_min, fpr_max ) :
    '''
    |    Return the indices corresponding to tpr min and max, fpr_min and fpr_max
    '''

    assert( tpr_min >= tpr[-1] and tpr_max <= tpr[0] ) 
    assert( fpr_min >= fpr[-1] and fpr_max <= fpr[0] ) 
    assert( tpr_min <= tpr_max and fpr_min <= fpr_max ) 

    tpr_indices = np.where( (tpr <= tpr_max) & (tpr >= tpr_min) )
    fpr_indices = np.where( (fpr <= fpr_max) & (fpr >= fpr_min) ) 

    return tpr_indices, fpr_indices

def get_filenames_in_threshold_range( trained_model, X, y, filenames, (tpr_min,tpr_max), (fpr_min, fpr_max) ) :
    '''
    |    Return the filenames and corresponding labels that satisfy
    |     the tpr or fpr range by broadcasting.
    '''
    
    filenames, scores, labels = get_ranked_predictions( trained_model, X, y, filenames )               
                   
    tpr, fpr = _calc_tpr_fpr( scores, labels )
    tpr_indices, fpr_indices = _get_tpr_fpr_indices( tpr, tpr_min, tpr_max, fpr, fpr_min, fpr_max ) 

    return zip( filenames[tpr_indices], labels[tpr_indices] ), \
                       zip( filenames[fpr_indices], labels[fpr_indices] )


