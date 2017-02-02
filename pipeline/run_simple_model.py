"""Simple logistic regression of a small number of variables.

Reads data from text files.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('Xtrain_file')
    parser.add_argument('ytrain_file')
    parser.add_argument('Xtest_file')
    parser.add_argument('yout_file')
    parser.add_argument('-C', '--logistic_regression_C', type=float,
                        required=False, default=10000.)
    parser.add_argument('-v', '--verbose', required=False,
                        action='store_true')

    args = vars(parser.parse_args())

    X_train = np.loadtxt(args['Xtrain_file'])
    y_train = np.loadtxt(args['ytrain_file'])

    model = LogisticRegression(C=args['logistic_regression_C'])
    model.fit(X_train, y_train)

    if args['verbose']:
        print "Accuracy on training set:", model.score(X_train, y_train)
        positive_idx = np.where(model.classes_ == 1.)[0][0]
        print "AUC on training set:",
        print roc_auc_score(y_train, model.predict_log_proba(X_train)[:, positive_idx])
        print "Confusion matrix on training set:"
        print confusion_matrix(y_train, model.predict(X_train))

    del X_train, y_train
    X_test = np.loadtxt(args['Xtest_file'])
    np.savetxt(args['yout_file'], model.predict(X_test))
