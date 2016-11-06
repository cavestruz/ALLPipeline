dir=$1

python test_trained_model.py $dir train > $dir/model_train.out
python test_trained_model.py -p $dir/ROC.pdf $dir test > $dir/model_test.out
