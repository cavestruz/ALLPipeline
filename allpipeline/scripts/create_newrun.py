import sys, os
import shutil
import argparse

parser=argparse.ArgumentParser(
    description='''This script creates one or more new runs for training and testing a model.''',
    epilog='''''',
    ) 

parser.add_argument('rundirs', type=str, nargs='+', 
                    help='location(s) of new rundirectory(ies) to create.  I recommend you create a separate directory for runs outside of this package')
parser.add_argument('-s', '--samplesdir', type=str, nargs=1, required=True, 
                    help='directory location of the config and pbs submission samples')
parser.add_argument('-o', '--overwrite', type=bool, nargs=1, default=False,
                    help='if a rundirectory exists, overwrite it.  Make sure to check the directory first.')
                    

args = parser.parse_args()


def create_run(rundir) :
    if args.overwrite : 
        shutil.rmtree(rundir)

    if os.path.exists(rundir) :
        print rundir, ' exists already.  Use the --overwrite argument if you wish to overwrite.'
        sys.exit()
    else :
        print 'making ', rundir
        os.makedirs(rundir)
    

def copy_sample_runfiles(rundir, samplesdir=None) :
    if samplesdir is None :
       samplesdir = args.samplesdir[0]
    try :
        assert(os.path.exists(samplesdir+'/sample_config.ini'))
    except AssertionError :
        print 'invalid argument for samplesdir ', args.samplesdir

    
    sample_config = samplesdir+'/sample_config.ini'
    sample_pbs = samplesdir+'/sample_pbs.sh'
    shutil.copy(sample_config, rundir)
    shutil.copy(sample_pbs, rundir)


if __name__ == '__main__' :
    for rundir in args.rundirs :
        create_run(rundir) 
        copy_sample_runfiles(rundir) 

