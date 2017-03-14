# Print all of the results: score and AUC as a function of C to a text file to plot

Cdir=$1

echo "C_logreg train_AUC test_AUC train_time test_time" > $Cdir/score_auc_C.txt 

for model in `ls -d $Cdir/model*`
do 

trainfile=$model/model_train.out
testfile=$model/model_test.out

C=$(cat $testfile | awk '/__C/ {print $NF}')
C=$(echo $C | rev | cut -d " " -f1 | rev)
#trainscore=$(cat $trainfile | awk '/Score/ {print $NF}')
trainauc=$(cat $trainfile | awk '/AUC/ {print $NF}')
#testscore=$(cat $testfile | awk '/Score/ {print $NF}')
testauc=$(cat $testfile | awk '/AUC/ {print $NF}')
traintime=$(cat $trainfile | awk '/Time/ {print $NF}')
echo $traintime
traintime=$(echo $traintime | cut -d " " -f1 )
echo $traintime
testtime=$(cat $testfile | awk '/Time/ {print $NF}')
echo "Traintime: " $traintime
echo $C $trainauc $testauc $traintime $testtime >> $Cdir/score_auc_C.txt 

done