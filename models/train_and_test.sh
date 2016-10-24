dir=$1

python generate_trained_model.py $dir > $dir/model_train.out
python test_trained_model.py $dir > $dir/model_test.out
