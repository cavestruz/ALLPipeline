from StrongCNN.utils import features_selection as fs
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt

datadir = sys.argv[1]
train_test_string = sys.argv[2]
if len(sys.argv) < 4 : pd_structure = True

else : pd_structure = False

def print_roc_to_file( avgscorefile, rocfile, figname, pd_structure ) :
    metric_data = pd.read_csv(avgscorefile,sep=' ')
    metric_data.sort('avgscores', inplace=True, ascending=False)
    print metric_data
    print metric_data['avgscores'][:]
    print metric_data['label'][:]
    avgscores = metric_data['avgscores'][:]
    labels = metric_data['label'][:]
    if pd_structure :
        fpr, tpr = fs._calc_tpr_fpr(avgscores,labels)

        metric_data['tpr'] = tpr
        metric_data['fpr'] = fpr

        metric_data.to_csv(path_or_buf=rocfile, sep=' ', index=False)
    else :
        print "Using metrics to calculate roc curve"        
        fpr, tpr, _ = roc_curve(labels, avgscores )

        np.savetxt(rocfile, np.array([tpr,fpr]).transpose(),header = 'tpr fpr')

    plt.plot(fpr, tpr)
    plt.savefig( figname )



if __name__ == '__main__' :

    print_roc_to_file( datadir+'/print_all_scores_for_debugging_'+train_test_string+'.txt', 
                       datadir+'/roc_'+train_test_string+'.txt', 
                       datadir+'/roc_curve_'+train_test_string+'.pdf',
                       pd_structure )
        
