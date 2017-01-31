import sys, os
from glob import glob
import pandas as pd

def print_ID_and_score( tprfile='tpr_filenames.txt', outfile='submission.txt' ) :
    df = pd.read_csv( tprfile, sep=' ') 
    # http://chrisalbon.com/python/pandas_regex_to_create_columns.html
    df['IDs'] = df['filename'].str.extract('(\d\d\d\d\d\d)')
    df.sort('IDs',inplace=True) 
    df.to_csv( path_or_buf=outfile, sep=' ', columns=['IDs', 'score'], index=False )

    
print_ID_and_score( tprfile='challenge_data/SpaceBased/tpr_filenames.txt' )
