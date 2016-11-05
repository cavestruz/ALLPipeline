import ast, sys, time
from StrongCNN.IO.config_parser import parse_configfile
from StrongCNN.IO.load_images import load_data
from StrongCNN.IO.augment_data import augment_data
from StrongCNN.utils.model_info import get_false_predictions_list
from _tools import generate_X_y, load_model

'''
Get the score and false ids of the model on either the training set or the test set
'''

cfgdir = sys.argv[1]
set_name = sys.argv[2]

cfg = parse_configfile(cfgdir)

start_time = time.time()

assert(set_name in ['test','train'])

# Collect testing data
X_test, y_test = load_data(cfg[set_name+'_filenames']['non_lens_glob'], 
                           cfg[set_name+'_filenames']['lens_glob'])


if 'augment_'+set_name+'_data' in cfg.keys() :
    X_test, y_test = augment_data( X_test, y_test, 
                                     cfg['augment_'+set_name+'_data']['method_label'],
                                     **ast.literal_eval(cfg['augment_'+set_name+'_data']['method_kwargs']))  
    
print "len(X_test) =", len(X_test)
print "len(y_test) =", len(y_test)

trained_model = load_model(cfgdir+'/'+cfg['model']['pklfile'])

X, y, filenames = generate_X_y(cfg[set_name+'_filenames']['non_lens_glob'], 
                                         cfg[set_name+'_filenames']['lens_glob']) 
print set_name+' filename glob', cfg[set_name+'_filenames']['non_lens_glob'], cfg[set_name+'_filenames']['lens_glob']
print ''
print 'Testing model parameter grid:'
for k,v in cfg['param_grid'].iteritems() :
    print k, v
    print ''

print 'False predictions: '
print get_false_predictions_list(trained_model, X, y, filenames)
print ''
print 'Time taken:', time.time() - start_time
