import ast, sys, time
from StrongCNN.IO.config_parser import parse_configfile
from StrongCNN.IO.load_images import load_data
from StrongCNN.IO.augment_data import augment_data
from StrongCNN.utils.model_info import get_false_predictions_list
from _tools import generate_X_y, load_model

cfgdir = sys.argv[1]
cfg = parse_configfile(cfgdir)

start_time = time.time()
# Collect training data
X_test, y_test = load_data(cfg['test_filenames']['non_lens_glob'], 
                           cfg['test_filenames']['lens_glob'])


if 'augment_test_data' in cfg.keys() :
    X_test, y_test = augment_data( X_test, y_test, 
                                     cfg['augment_test_data']['method_label'],
                                     **ast.literal_eval(cfg['augment_test_data']['method_kwargs']))  
    
print "len(X_test) =", len(X_test)
print "len(y_test) =", len(y_test)

trained_model = load_model(cfgdir+'/'+cfg['model']['pklfile'])

X, y, filenames = generate_X_y(cfg['train_filenames']['non_lens_glob'], 
                                         cfg['train_filenames']['lens_glob']) 
print 'Test glob', cfg['test_filenames']['non_lens_glob'], cfg['test_filenames']['lens_glob']
print ''
print 'Testing model parameter grid:'
for k,v in cfg['param_grid'].iteritems() :
    print cfg['param_grid'][k], v
    print ''

print get_false_predictions_list(trained_model, X, y, filenames)

print 'Time taken:', time.time() - start_time
