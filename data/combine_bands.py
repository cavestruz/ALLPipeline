from astropy.io.fits import getdata, writeto
import numpy as np
from glob import glob
import os
                         
def get_file_by_ID( directory, ID ) :
    filename = glob(directory+'/*'+ID+'*.fits*')
    assert( len(filename) == 1 )
    return filename[0] 

def get_all_bands( band_directories, ID ) :
    return np.array([ getdata( get_file_by_ID( band_dir, ID ) ) for \
                              for band_dir in band_directories ])

def write_combined_fits( combined_directory, band_directories, ID ) :

    writeto( combined_directory+'/imageSDSS_RIGU-'+ID+'.fits',
             get_all_bands( band_directories, ID ) )

def get_IDs( IDs_directory, IDs_glob='*' ) :
    filenames = glob(IDs_directory+'/'+IDs_glob+'.fits')
    return [ filename.split('/')[-1].split('-')[-1].split('.fits')[0] \
                 for filename in filenames ]

def write_combined_fits_for_IDs( IDs_directory, IDs_glob, band_directories, combined_directory ) :
    if not os.path.exists(combined_directory) :
        printing "making ", combined_directory
        os.mkdir( combined_directory ) 
    for ID in get_IDs( IDs_directory, IDs_glob=IDs_glob ) :
        write_combined_fits(  combined_directory, band_directories, ID  )


if __name__ == "__main__" :
    write_combined_fits_for_IDs( 
        'GroundBased/lensed-Band1/',
        '*10000*',
        ['GroundBased/lensed-Band'+str(i)+'/' for i in range(1,5) ],
        'GroundBased/lensed-all/'
        )


