dir=$1

python test_trained_model.py -s $dir/filenames_scores.txt -T time $dir test >> $dir/model_test.out
python ../data/print_avg_rotated_scores.py $dir test

