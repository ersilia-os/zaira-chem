FILENAME=$1

if [ $# -eq 0 ] ; then
    echo "Filename not supplied."
else

python do_training.py --filename $FILENAME --verbose True

fi