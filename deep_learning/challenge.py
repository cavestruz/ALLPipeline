"""Use a convolutional neural network on
the challenge data."""
import glob
import csv
import numpy as np
import random

import tflearn
from astropy.io import fits as pyfits
from sklearn import metrics
from conv_net import conv_network

def load_images_labels(filenames, name_prefix, name_suffix,
                       start_num, end_num, label_file, vrb=False):
    filenumbers = [int(filename[len(name_prefix):-len(name_suffix)])
                   for filename in filenames]
    filtered_files = [(number, name) for number, name
                      in zip(filenumbers, filenames)
                      if start_num <= number <= end_num]

    images = np.array([np.array(pyfits.getdata(name))
                       for number, name in filtered_files])
    image_dim = images.shape[1:]
    images = images.reshape((-1,) + image_dim + (1,))
    
    if vrb:
        print "Read", images.shape[0], "images with dimensions", image_dim

    labels = {int(row['ID']): int(row['is_lens'])
              for row in csv.DictReader(open(label_file))}
    labels = [labels[number] for number, name in filtered_files]
    assert set(labels) == {0, 1}, set(labels)
    labels = np.array([[0, 1] if label else [1, 0]
                       for label in labels])

    assert labels.shape[0] == images.shape[0], (labels.shape[0], images.shape[0])

    if vrb:
        print labels[:,1].sum(), "lenses found,", labels[:,0].sum(), "non-lenses found"

    return images, labels

def train_test_split(X, y, test_size, vrb=False):
    indices = range(X.shape[0])
    random.shuffle(indices)
    num_test = int(test_size * X.shape[0])
    X_test = X[indices[:num_test], :]
    X_train = X[indices[num_test:], :]
    y_test = y[indices[:num_test], :]
    y_train = y[indices[num_test:], :]

    if vrb:
        print "X_train.shape =", X_train.shape
        print "X_test.shape =", X_test.shape
        print "y_train.shape =", y_train.shape
        print "y_test.shape =", y_test.shape

    return X_train, X_test, y_train, y_test
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file_prefix')
    parser.add_argument('start_num', type=int)
    parser.add_argument('end_num', type=int)
    parser.add_argument('label_file')
    parser.add_argument('n_epoch', type=int)
    args = parser.parse_args()

    filenames = glob.glob(args.image_file_prefix + '*.fits')
    X, y = load_images_labels(filenames, args.image_file_prefix, '.fits',
                              args.start_num, args.end_num, args.label_file,
                              vrb=True)
    print
    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size = 0.2, vrb=True)
    del X, y

    network = conv_network(X_train.shape[1:3], 2)
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': X_train}, {'target' : y_train},
              n_epoch=args.n_epoch,
              validation_set=({'input': X_test}, {'target': y_test}),
              snapshot_step=100, show_metric=True,
              run_id='convnet_challenge')

    train_probability = np.array(model.predict(X_train))[:,1]
    test_probability = np.array(model.predict(X_test))[:,1]

    train_predict = (train_probability > 0.5).astype(int)
    test_predict = (test_probability > 0.5).astype(int)

    train_score = metrics.accuracy_score(y_train[:,1], train_predict)
    train_confusion_matrix = metrics.confusion_matrix(y_train[:,1], train_predict)
    train_auc = metrics.roc_auc_score(y_train[:,1], train_probability)

    test_score = metrics.accuracy_score(y_test[:,1], test_predict)
    test_confusion_matrix = metrics.confusion_matrix(y_test[:,1], test_predict)
    test_auc = metrics.roc_auc_score(y_test[:,1], test_probability)

    print 'Train Accuracy: {0:f}'.format(train_score)
    print 'Test Accuracy: {0:f}'.format(test_score)
    print
    print 'Train Confusion Matrix:'
    print train_confusion_matrix
    print
    print 'Test Confusion Matrix:'
    print test_confusion_matrix
    print
    print 'Train AUC: {0:f}'.format(train_auc)
    print 'Test AUC: {0:f}'.format(test_auc)
