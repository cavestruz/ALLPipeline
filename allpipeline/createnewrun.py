import sys,os
import shutil

dirs = str(sys.argv[1:2]).replace("['","").replace("']","")

if os.path.exists(dirs):
    print "exists!"
else: 
    os.makedirs(dirs)

sample_config = './samples/sample_config.ini'
sample_pbs = './samples/sample_pbs.sh'
shutil.copy(sample_config, dirs)
shutil.copy(sample_pbs, dirs)
