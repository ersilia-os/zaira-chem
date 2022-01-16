FILENAME=$1

if [ $# -eq 0 ] ; then
    echo "Filename not supplied."
else

python do_novo.py --filename $FILENAME --verbose True

fi
