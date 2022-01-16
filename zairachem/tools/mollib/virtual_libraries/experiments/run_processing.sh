FILENAME=$1

if [ $# -eq 0 ] ; then
    echo "Filename not supplied."
else

python do_data_processing.py --filename $FILENAME --verbose True

fi