import sys
from sklearn.cross_validation import train_test_split
from StrongCNN.IO.config_parser import parse_configfile
from StrongCNN.IO.load_images import load_data
from StrongCNN.IO.augment_data import augment_methods, augment_data
from StrongCNN.grid_searches.tools import build_parameter_grid, grid_search
from StrongCNN.pipeline.build_pipeline import build_pipeline
import ast

cfg = parse_configfile(sys.argv[1])

# Collect training data
X_train, y_train = load_data(cfg['filenames']['non_lens_glob'], 
                             cfg['filenames']['lens_glob'])


if 'augment_train_data' in cfg.keys() :
    X_train, y_train = augment_data( X_train, y_train, 
                                     cfg['augment_train_data']['method_label'],
                                     **ast.literal_eval(cfg['augment_train_data']['method_kwargs']))  
    
print "len(X_train) =", len(X_train)
print "len(y_train) =", len(y_train)

# Build the pipeline
pipeline = build_pipeline(cfg['image_processing'].values(), 
                          cfg['classifier']['label'])

# Perform the grid search
grid_search(pipeline, param_grid, X_train, y_train, X_test, y_test) 
