#PBS -q batch
#PBS -N grid_search
#PBS -l nodes=1:ppn=8,mem=35gb
#PBS -l walltime=12:00:00
#PBS -F arguments
#PBS -o grid_search.out
#PBS -e grid_search.err


cd $PBS_O_WORKDIR

#########################################################
#   Submission script to do grid search#
#   Usage on PBS queuing system#
#   Need config.ini in PBS_O_WORKDIR for this to work#
#########################################################

python model_grid_search.py $PBS_O_WORKDIR