# From the tpr_fpr filenames file, print the lens model parameters in
# separate columns.  Filenames look like:
# DATAdir/lensed_<index>_<veldisp>_<ellip>_<angle>_<z>_<mag>_imgs.fits

Cdir=$1

echo "# index veldisp ell angle z mag score tpr fpr" > $Cdir/control_test_params.txt 

tprfile=$model/filenames_tpr_fpr.out # Check name of this...

# Split each line of the tpr file by filename, score, tpr, and fpr
$(cat tprfile | awk '//')
# Split each line of the tpr file by the last few _

testauc=$(cat $testfile | awk '/AUC/ {print $NF}')

echo $testauc >> $Cdir/control_test_params.txt

