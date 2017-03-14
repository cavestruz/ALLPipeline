import sys, os
from glob import glob
import pandas as pd
from StrongCNN.utils.model_info import generate_AUC_fpr_tpr

scoresdir = sys.argv[1] 
set_name = sys.argv[2]

def print_ID_and_avg_score( fdir='.', 
                            outfile1='/avestruz_submission_avg_scores_ordered_by_id_'+set_name+'.txt', 
                            outfile2='/avestruz_submission_avg_scores_ordered_by_score_'+set_name+'.txt', 
                            outfile3='/print_all_scores_for_debugging_'+set_name+'.txt',
                            AUC_file='/rotated_aucs_'+set_name+'.txt',
                            IDs_regex='\d\d\d\d\d\d') :
    '''Create four dataframes, order by id, create new dataframe and print'''

    df = {}
    rotated_scores = glob(fdir+'/filenames_scores_'+set_name+'?.txt')

    for rotation in rotated_scores :
        df[rotation] = pd.read_csv( rotation, sep=' ' )
        print df[rotation]
        df[rotation]['IDs'] = df[rotation]['filename'].str.extract(IDs_regex)
        df[rotation].sort_values(by='IDs',inplace=True) 
    
    # append other scores to first
    df[rotated_scores[0]]['avgscores'] = 0.
    for i, rotation in enumerate(rotated_scores) :
        df[rotated_scores[0]]['avgscores'] += df[rotation]['score']/4.
        df[rotated_scores[0]]['score'+str(i)] = df[rotation]['score']

    df[rotated_scores[0]].to_csv( path_or_buf=fdir+outfile1, sep=' ', columns=['IDs', 'avgscores'], index=False )
    df[rotated_scores[0]].to_csv( path_or_buf=fdir+outfile3, sep=' ', columns=['IDs', 'label', 'avgscores', 'score0', 'score1', 'score2', 'score3'],
                                  index=False )

    if AUC_file != None :
        f = open(fdir+AUC_file, 'w')
        avg_AUC, _ = generate_AUC_fpr_tpr( df[rotated_scores[0]]['label'], df[rotated_scores[0]]['avgscores'] )
        f.write('%1.6f\n' % avg_AUC )
        for rotation in rotated_scores :
            AUC, _ = generate_AUC_fpr_tpr( df[rotation]['label'], df[rotation]['score'] )
            f.write('%1.6f\n' % AUC )

    df[rotated_scores[0]].sort_values(by='avgscores',inplace=True) 
    df[rotated_scores[0]].to_csv( path_or_buf=fdir+outfile2, sep=' ', columns=['IDs', 'scores'], index=False )

print_ID_and_avg_score( fdir=scoresdir,
                        IDs_regex='lensed\_(\d+)\_'
                        )
