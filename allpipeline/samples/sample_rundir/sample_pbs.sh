#PBS -q batch
#PBS -N train
#PBS -l nodes=1:ppn=8,mem=35gb
#PBS -l walltime=12:00:00
#PBS -F arguments
#PBS -o model_train.out
#PBS -e model_train.err


cd $PBS_O_WORKDIR

#########################################################
#   Submission script to train data			#
#   Usage on PBS queuing system				#
#   Need config.ini in PBS_O_WORKDIR for this to work	#
#########################################################

python $(pwd)/../models/generate_trained_model.py $PBS_O_WORKDIR 

