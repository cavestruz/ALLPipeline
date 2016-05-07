import numpy as np
import glob
from sklearn.pipeline import Pipeline
import image_processing


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('non_lens_glob')
    parser.add_argument('lens_glob')

    args = vars(parser.parse_args())

    non_lens_filenames = glob.glob(args['non_lens_glob'])
    lens_filenames = glob.glob(args['lens_glob'])
    filenames = non_lens_filenames + lens_filenames
    y = [0] * len(non_lens_filenames) + [1] * len(lens_filenames)
    
    estimators = [('load_images', image_processing.LoadImages()),
                  ('median_smooth', image_processing.MedianSmooth(5)),
                  ('hog', image_processing.HOG(orientations = 8,
                                               pixels_per_cell = (16, 16),
                                               cells_per_block = (1, 1)))]
    pipeline = Pipeline(estimators)

    X = pipeline.transform(filenames)
    print X.shape
    import cPickle as pickle
    pickle.dump(X, open('X.pkl', 'w'))

