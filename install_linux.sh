WORKDIR=$PWD

conda init bash

# create zairachem conda environment
ZAIRACHEM_ENVIRONMENT='zairachem'
conda create -n $ZAIRACHEM_ENVIRONMENT python=3.7 -y
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate $ZAIRACHEM_ENVIRONMENT

# pip
python3 -m pip install -U pip
python3 -m pip install -U setuptools wheel

# other pip-installable dependencies
pip install autokeras==1.0.16

# install autogluon cpu
python3 -m pip install -U "mxnet<2.0.0"
python3 -m pip install autogluon

# install autogluon gpu
# Here we assume CUDA 10.1 is installed.  You should change the number
# according to your own CUDA version (e.g. mxnet_cu100 for CUDA 10.0).
#python3 -m pip install -U "mxnet_cu101<2.0.0"
#python3 -m pip install autogluon


python3 -m pip install "xgboost==1.3.3"
python3 -m pip install "SQLAlchemy<1.4.0"

# install zairachem
conda install -c conda-forge fpsim2 -y
pip install -q -U keras-tuner

# install ersilia
python3 -m pip install ersilia
ersilia --help

# install isaura
python3 -m pip install git+https://github.com/ersilia-os/isaura.git

# install stylia
python3 -m pip install git+https://github.com/ersilia-os/stylia.git

# install lazy-qsar
python3 -m pip install git+https://github.com/ersilia-os/lazy-qsar.git

# install zairachem
python -m pip install -e .

# create extra conda envirnoments

# install molmap
MOLMAP_ENVIRONMENT='molmap'
conda create -n $MOLMAP_ENVIRONMENT python=3.6 -y
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate $MOLMAP_ENVIRONMENT
cd zairachem/tools/molmap/bidd-molmap/
conda install -c tmap tmap -y
conda install -c conda-forge rdkit=2020.03 -y
python -m pip install -r requirements.txt
python -m pip install h5py==2.10.0
cd $WORKDIR

# install mollib
cd zairachem/tools/mollib/virtual_libraries/
bash install_linux.sh
cd $WORKDIR