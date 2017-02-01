import sys, os
from glob import glob
import pandas as pd
from StrongCNN.utils.model_info import generate_AUC_fpr_tpr

scoresdir = sys.argv[1] 
# sub_type = sys.argv[2]

def print_ID_and_avg_score( fdir='.', outfile='/avg_rotated_scores.txt', AUC_file='/rotated_aucs.txt' ) :
    '''Create four dataframes, order by id, create new dataframe and print'''

    df = {}
    rotated_scores = glob(fdir+'/filenames_scores?.txt')

    for rotation in rotated_scores :
        df[rotation] = pd.read_csv( rotation, sep=' ' )
        print df[rotation]
        df[rotation]['IDs'] = df[rotation]['filename'].str.extract('(\d\d\d\d\d\d)')
        df[rotation].sort_values(by='IDs',inplace=True) 

    # append other scores to first
    df[rotated_scores[0]]['avgscores'] = df[rotated_scores[0]]['score']/4.
    for rotation in rotated_scores[1:] :
        df[rotated_scores[0]]['avgscores'] += df[rotation]['score']/4.

    df[rotated_scores[0]].to_csv( path_or_buf=fdir+outfile, sep=' ', columns=['IDs', 'score'], index=False )

    if AUC_file != None :
        f = open(fdir+AUC_file, 'w')
        avg_AUC, _ = generate_AUC_fpr_tpr( df[rotated_scores[0]]['label'], df[rotated_scores[0]]['avgscores'] )
        f.write('%1.6f\n' % avg_AUC )
        for rotation in rotated_scores :
            AUC, _ = generate_AUC_fpr_tpr( df[rotation]['label'], df[rotation]['score'] )
            f.write('%1.6f\n' % AUC )


print_ID_and_avg_score( fdir=scoresdir,
                        )
