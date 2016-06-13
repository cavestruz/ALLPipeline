import numpy as np
import glob
from sklearn.pipeline import Pipeline
import image_processing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter

def confusion_matrix(predicted, actual):
    predicted, actual = map(np.array, [predicted, actual])
    return Counter(zip(predicted, actual))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('non_lens_glob')
    parser.add_argument('lens_glob')

    args = vars(parser.parse_args())
    
    # Load the data. X is a list of numpy arrays
    # which are the images.
    non_lens_filenames = glob.glob(args['non_lens_glob'])
    lens_filenames = glob.glob(args['lens_glob'])
    filenames = non_lens_filenames + lens_filenames
    X = image_processing.load_images(filenames)
    y = [0] * len(non_lens_filenames) + [1] * len(lens_filenames)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
    print "len(X_train) =", len(X_train)
    print "len(y_train) =", len(y_train)
    print "len(X_test) =", len(X_test)
    print "len(y_test) =", len(y_test)
    print

    # Create the pipeline which consists of image
    # processing and a classifier
    image_processors = [('median_smooth', image_processing.MedianSmooth(5)),
                        ('hog', image_processing.HOG(orientations = 8,
                                                     pixels_per_cell = (16, 16),
                                                     cells_per_block = (1, 1)))]
    classifier = ('logistic_regression', LogisticRegression())
    estimators = image_processors + [classifier]
    
    pipeline = Pipeline(estimators)

    # Train the model on the training set
    pipeline.fit(X_train, y_train)
    print "Confusion matrix on training set"
    print confusion_matrix(pipeline.predict(X_train), y_train)
    print
    print "Score on training set =", pipeline.score(X_train, y_train)
    print

    # Score the test set
    print "Confusion matrix on test set"
    print confusion_matrix(pipeline.predict(X_test), y_test)
    print
    print "Score on test set =", pipeline.score(X_test, y_test)
