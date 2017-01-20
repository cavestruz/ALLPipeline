"""Use a convolutional neural network on
the challenge data."""
import glob
import csv
import numpy as np
import random

from astropy.io import fits as pyfits
from sklearn import metrics
from tensorflow.contrib import learn as skflow
from conv_net import conv_model

def train_test_split(X, y, test_size):
    indices = range(X.shape[0])
    random.shuffle(indices)
    num_test = int(test_size * X.shape[0])
    X_test = X[indices[:num_test], :]
    X_train = X[indices[num_test:], :]
    y_test = y[indices[:num_test]]
    y_train = y[indices[num_test:]]

    return X_train, X_test, y_train, y_test
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file_prefix')
    parser.add_argument('start_num', type=int)
    parser.add_argument('end_num', type=int)
    parser.add_argument('label_file')
    parser.add_argument('num_steps', type=int)
    args = parser.parse_args()

    filenames = glob.glob(args.image_file_prefix + '*.fits')
    filenumbers = [int(filename[len(args.image_file_prefix):-len('.fits')])
                   for filename in filenames]
    filtered_files = [(number, name) for number, name
                      in zip(filenumbers, filenames)
                      if args.start_num <= number <= args.end_num]

    images = np.array([np.array(pyfits.getdata(name)) for number, name in filtered_files])
    image_dim = images.shape[1:]
    images = images.reshape(images.shape[0], image_dim[0] * image_dim[1])

    print "Read", images.shape[0], "images with dimensions", image_dim
    
    labels = {int(row['ID']): int(row['is_lens'])
              for row in csv.DictReader(open(args.label_file))}
    labels = np.array([labels[number] for number, name in filtered_files])

    assert labels.shape[0] == images.shape[0], (labels.shape[0], images.shape[0])
    assert len(labels.shape) == 1, len(labels.shape)
    assert set(labels) == {0, 1}, set(labels)

    print labels.sum(), "lenses found,", labels.shape[0] - labels.sum(), "non-lenses found"
    
    X_train, X_test, y_train, y_test \
        = train_test_split(images, labels, test_size = 0.2)

    print "X_train.shape =", X_train.shape
    print "X_test.shape =", X_test.shape
    print "y_train.shape =", y_train.shape
    print "y_test.shape =", y_test.shape

    model_fn \
        = lambda feature, target, mode : conv_model(feature, target, mode,
                                                    image_dim=image_dim,
                                                    layer1_size=32,
                                                    layer2_size=64,
                                                    dense_layer_size=1024)
    
    classifier = skflow.Estimator(model_fn=model_fn)
    classifier.fit(X_train, y_train,
                   batch_size = 100,
                   steps = args.num_steps)
    train_score \
        = metrics.accuracy_score(y_train,
                                 list(classifier.predict(X_train)))
    train_confusion_matrix \
        = metrics.confusion_matrix(y_train,
                                   list(classifier.predict(X_train)))
    test_score \
        = metrics.accuracy_score(y_test,
                                 list(classifier.predict(X_test)))
    test_confusion_matrix \
        = metrics.confusion_matrix(y_test,
                                   list(classifier.predict(X_test)))

    print 'Train Accuracy: {0:f}'.format(train_score)
    print 'Test Accuracy: {0:f}'.format(test_score)
    print
    print 'Train Confusion Matrix:'
    print train_confusion_matrix
    print
    print 'Test Confusion Matrix:'
    print test_confusion_matrix
