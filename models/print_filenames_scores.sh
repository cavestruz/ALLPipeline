dir=$1

python test_trained_model.py -s $dir/filenames_scores.txt $dir test >> $dir/model_test.out

