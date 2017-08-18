from astropy.io.fits import getdata, writeto
import numpy as np
from glob import glob
import os, time, sys
from multiprocessing import Pool

time_now = time.time()               

# To identify unique ids
# IDs_dir='/data/avestruz/StrongCNN/Challenge/FinalData/GroundBased/'
# IDs_glob=['*[12]*', '*[34]*', '*[56]', '*[78]*', '*[90]*' ]
# band_dir = ['/data/avestruz/StrongCNN/Challenge/FinalData/GroundBased/'+\
#                 '-Band'+str(i)+'/' for i in range(1,5) ]
combined_dir = '/data/avestruz/StrongCNN/Challenge/FinalData/GroundBased/combined_bands'


# IDs_glob=['*100?[12]*', '*100?[34]*', '*100?[56]', '*100?[78]*', '*100?[90]*' ]
# band_dir = ['~/Downloads/Data_KiDS_Big.*/Public/Band'+str(i)+'/' for i in range(1,5) ]
# IDs_dir=band_dir[0]

# IDs_glob=['/home/babyostrich/Downloads/Data_KiDS_Big.'+str(i)+'/Public/Band1/' for i in range(0,11)]
#combined_dir = '/home/babyostrich/Documents/Repos/StrongCNN/data/FinalChallengeData/GroundBasedData/combined_data'

          
def get_file_by_ID( directory, ID ) :
    filename = glob(directory+'/*'+ID+'*.fits*')
    try :
        assert( len(filename) == 1 )
    except AssertionError : print directory, ID, filename
    return filename[0] 

def get_all_bands( band_directories, ID ) :
    return np.array([ getdata( get_file_by_ID( band_dir, ID ) )  \
                          for band_dir in band_directories ])

def write_combined_fits( combined_directory, band_directories, ID ) :
    writeto( combined_directory+'/imageSDSS_RIGU-'+ID+'.fits',
             get_all_bands( band_directories, ID ) )

def get_IDs( IDs_directory, IDs_glob='*' ) :
    filenames = glob(IDs_directory+'/'+IDs_glob+'.fits')
    print "Number of ids in this data subset", len(filenames)
    print "first from ", filenames[0]
    return [ filename.split('/')[-1].split('-')[-1].split('.fits')[0] \
                 for filename in filenames ]

def write_combined_fits_for_IDs( i, IDs_glob='*', IDs_directory=None, band_directories=None, combined_directory=combined_dir ) :
    print "Data subset:", i
    band_directories = [ '/data/avestruz/StrongCNN/Challenge/FinalData/GroundBased/Data_KiDS_Big.'+str(i)+'/Public/Band'+str(j)+'/' for j in range(1,5) ]
    IDs_directory = band_directories[0]

    if not os.path.exists(combined_directory) :
        print "making ", combined_directory
        os.mkdir( combined_directory ) 
    IDs = get_IDs( IDs_directory ) 
    print "num ids from get_IDs ", len(IDs)
    for ID in IDs :
        write_combined_fits(  combined_directory, band_directories, ID  )


if __name__ == "__main__" :
    #p = Pool(4) 

    #p.map(write_combined_fits_for_IDs, range(4))
    for i in [0,1,2,3] : 
        write_combined_fits_for_IDs(i)
    print "Time taken for ",  time.time() - time_now

