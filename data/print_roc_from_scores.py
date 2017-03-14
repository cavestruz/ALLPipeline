from StrongCNN.utils import features_selection as fs
import pandas as pd
import sys

datadir = sys.argv[1]
train_test_string = sys.argv[2]

def print_roc_to_file( avgscorefile, rocfile ) :
    metric_data = pd.read_csv(avgscorefile,sep=' ')
    metric_data.sort('avgscores', inplace=True, ascending=True)
    print metric_data
    print metric_data['avgscores'][:]
    print metric_data['label'][:]
    fpr, tpr = fs._calc_tpr_fpr(metric_data['avgscores'][:],metric_data['label'][:])

    metric_data['tpr'] = tpr
    metric_data['fpr'] = fpr

    metric_data.to_csv(path_or_buf=rocfile, sep=' ',columns=['tpr','fpr'], index=False)


if __name__ == '__main__' :
    print_roc_to_file( datadir+'/print_all_scores_for_debugging_'+train_test_string+'.txt', datadir+'/roc_'+train_test_string+'.txt' )
