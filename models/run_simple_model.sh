cfgdir=$1
fileglob=$2


#python run_simple_model.py --cfgdir $cfgdir -C 10000 -p pickle $fileglob 
python run_simple_model.py --cfgdir $cfgdir -C 10000 -t test $fileglob >> $cfgdir/out.txt