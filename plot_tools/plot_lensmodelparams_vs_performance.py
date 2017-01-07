import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

testdir = sys.argv[1]
paramfile = testdir+'/control_test_params.txt'
control_test = pd.read_csv( paramfile, delim_whitespace=True )

lens_model_params = ['veldisp', 'ellipticity', 'orientationangle', 
                     'z', 'magnitude']

def plot_ROC() : 
    plt.plot(control_test['fpr'], control_test['tpr'],'b')
    plt.plot(np.arange(0,1.1,.1),np.arange(0,1.1,.1), 'g:')
    plt.xlabel('False Positive Rate',fontsize='xx-large')
    plt.ylabel('True Positive Rate',fontsize='xx-large')
    plt.savefig(testdir+'/ROC_curve.pdf')


def plot_score_dependence() :
    lensed = control_test[control_test['label'] == 1]
    unlensed = control_test[control_test['label'] == 0]
    for key in control_test.keys() :
        if key not in lens_model_params :
        for label,color in zip([lensed, unlensed],['g','r']) :
            plt.plot( label[key], label['score'], color+'*', lw=0.01)
            plt.xlabel(key.replace('_',' '),fontsize='xx-large')
            plt.ylabel('Score',fontsize='xx-large')
            plt.yscale('log')
            plt.savefig(testdir+'/score_vs_'+key+'.pdf')

def plot_colorcoded_score_dependence(colormap=cm.hot) :
    lensed = control_test[control_test['label'] == 1]
    unlensed = control_test[control_test['label'] == 0]
    for key in control_test.keys() :
        if key not in lens_model_params :
            next
        for color_key in lens_model_params :
            if color_key == key : next
            for label,marker in zip([lensed, unlensed],['*','o']) :
                label[color_key+'_color'] = colormap(label[color_key])
                plt.plot( label[key], label['score'], 
                          color=label[color_key+'_color'],ms=marker,lw=0.01)
                plt.xlabel(key.replace('_',' '),fontsize='xx-large')
                plt.ylabel('Score',fontsize='xx-large')
                plt.yscale('log')
                plt.savefig(testdir+'/score_vs_'+key+'_colorcodedby_'+color_key+'.pdf')
