MOLMAP_ENVIRONMENT='molmap'
conda create -n $MOLMAP_ENVIRONMENT python=3.6 -y
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate $MOLMAP_ENVIRONMENT
conda install -c tmap tmap -y
cd zairachem/tools/molmap/bidd-molmap/
python -m pip install -r requirements.txt
cd $WORKDIR