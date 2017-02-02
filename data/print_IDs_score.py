import sys, os
from glob import glob
import pandas as pd

scoresdir = sys.argv[1] 
sub_type = sys.argv[2]

assert( sub_type == 'space' or sub_type == 'ground' )

def print_ID_and_score( tprfile='tpr_filenames.txt', outfile1='avestruz_space_submission_ordered_by_id.txt', outfile2='avestruz_space_submission_ordered_by_score.txt' ) :
    df = pd.read_csv( tprfile, sep=' ') 
    # http://chrisalbon.com/python/pandas_regex_to_create_columns.html
    df['IDs'] = df['filename'].str.extract('(\d\d\d\d\d\d)')
    df.sort_values(by='score',inplace=True)
    df.to_csv( path_or_buf=outfile2, sep=' ', columns=['IDs', 'score'], index=False )
    df.sort_values(by='IDs',inplace=True) 
    df.to_csv( path_or_buf=outfile1, sep=' ', columns=['IDs', 'score'], index=False )

    
print_ID_and_score( tprfile=scoresdir+'/filenames_scores.txt',
                    outfile1=scoresdir+'/avestruz_'+sub_type+'_submission_ordered_by_id.txt',
                    outfile2=scoresdir+'/avestruz_'+sub_type+'_submission_ordered_by_score.txt')
