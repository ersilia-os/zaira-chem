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
python3 -m pip install tables openpyxl

# other pip-installable dependencies
python3 -m pip install tensorflow==2.10.0
python3 -m pip install autokeras==1.0.20 #==1.0.16

# install autogluon cpu
python3 -m pip install -U "mxnet<2.0.0"
python3 -m pip install autogluon.tabular[all]==0.5.2

# install autogluon gpu
# Here we assume CUDA 10.1 is installed.  You should change the number
# according to your own CUDA version (e.g. mxnet_cu100 for CUDA 10.0).
#python3 -m pip install -U "mxnet_cu101<2.0.0"
#python3 -m pip install autogluon

# necessary requirements for flaml
python3 -m pip install "xgboost==1.3.3"
python3 -m pip install "SQLAlchemy<1.4.0"

# install extra dependencies
python3 -m pip install git+https://github.com/chembl/FPSim2.git@0.3.0
python3 -m pip install -q -U keras-tuner==1.1.3

# install ersilia
python3 -m pip install git+https://github.com/ersilia-os/ersilia.git
ersilia --help

# install ersilia compound embedding
python3 -m pip install git+https://github.com/ersilia-os/compound-embedding-lite.git

# install isaura
python3 -m pip install git+https://github.com/ersilia-os/isaura.git@ce293244ad0bdd6d7d4f796d2a84b17208a87b56

# install stylia
python3 -m pip install git+https://github.com/ersilia-os/stylia.git

# install lazy-qsar
python3 -m pip install git+https://github.com/ersilia-os/lazy-qsar.git@0.1

# install melloddy-tuner
python3 -m pip install git+https://github.com/melloddy/MELLODDY-TUNER.git@2.1.3

# install tabpfn
python3 -m pip install tabpfn==0.1.8

# install imblearn
python3 -m pip install imbalanced-learn==0.10.1

# install zairachem
python3 -m pip install -e .
