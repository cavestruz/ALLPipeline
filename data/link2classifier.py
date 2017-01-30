import sys, os
from glob import glob
import pandas as pd

def collect_ids_by_classification(csvfile='classifications.csv') :
    classifications = pd.read_csv(csvfile, delimiter=',')
    lensed_ids = classifications['ID'][classifications['is_lens']==1]
    unlensed_ids = classifications['ID'][classifications['is_lens']==0]
    
    return [lensed_ids, unlensed_ids]

def create_links(datadir='{0,1}/Data.0/Public/Band1/',lensdir='./lensed/',unlensdir='./unlensed/',csvfile='classifications.csv') : 
    for ids, ldir in zip(collect_ids_by_classification(csvfile),[lensdir,unlensdir]) :
        if not os.path.exists(ldir) :
            print "making "+ldir
            os.mkdir(ldir)
        for i in ids[10:] :
            #print datadir+"*"+str(i)+"*.fits"
            fitsfile = glob(datadir+"*"+str(i)+"*.fits")[0]
            # Create hard link
            #print "linking "+fitsfile+" to "+ldir
            os.system('ln -s '+fitsfile+' '+ldir)

#create_links(datadir="dummy_data/")
#create_links(datadir="/data/avestruz/StrongCNN/Challenge/SpaceBased/SpaceBasedTraining/[01]/Data.[01]/Public/Band1/*")

# Create soft links for NoSourceImage
create_links(datadir="/data/avestruz/StrongCNN/Challenge/SpaceBased/SpaceBasedTraining/[01]/Data.[01]/Private/Band1/NoSourceImage/*",
             lensdir="./NoSourceImageLensed/",
             unlensdir="./NoSourceImageUnlensed/")
# Create soft links for NoLensImage
create_links(datadir="/data/avestruz/StrongCNN/Challenge/SpaceBased/SpaceBasedTraining/[01]/Data.[01]/Private/Band1/NoLensImage/*",
             lensdir="./NoLensImageLensed/",
             unlensdir="./NoLensImageUnlensed/")
