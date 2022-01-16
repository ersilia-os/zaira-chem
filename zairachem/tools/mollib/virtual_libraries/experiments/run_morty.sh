FILENAME=$1

if [ $# -eq 0 ] ; then
    echo "Filename not supplied."
else

bash run_processing.sh $FILENAME &&
bash run_training.sh $FILENAME &&
bash run_generation.sh $FILENAME &&
bash run_analysis.sh $FILENAME

fi
