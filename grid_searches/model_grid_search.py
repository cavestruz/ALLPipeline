import sys
from sklearn.cross_validation import train_test_split
from StrongCNN.IO.config_parser import parse_configfile
from StrongCNN.IO.load_images import load_data
from StrongCNN.grid_searches.tools import build_parameter_grid, grid_search
from StrongCNN.pipeline.build_pipeline import build_pipeline


if __name__ == "__main__":

    cfg = parse_configfile(sys.argv[1])

    X_train, X_test, y_train, y_test = \
        train_test_split( *load_data(cfg['filenames']['non_lens_glob'], 
                                     cfg['filenames']['lens_glob']), 
                           train_size=0.2 )

    
    print "len(X_train) =", len(X_train)
    print "len(y_train) =", len(y_train)
    print "len(X_test) =", len(X_test)
    print "len(y_test) =", len(y_test)
    print


    param_grid = build_parameter_grid(cfg['param_grid'])

    pipeline = build_pipeline(cfg['image_processing'].values(), 
                              cfg['classifier']['label'])

    grid_search(pipeline, param_grid, X_train, y_train, X_test, y_test) 
