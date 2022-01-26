WORKDIR=$PWD

# create zairachem conda environment
ZAIRACHEM_ENVIRONMENT='zairachem_test'
conda create -n $ZAIRACHEM_ENVIRONMENT python=3.7
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate $ZAIRACHEM_ENVIRONMENT

# install ersilia
python -m python -m pip install git+git@github.com:ersilia-os/ersilia.git
ersilia --help

# install isaura
python -m python -m pip install git+git@github.com:ersilia-os/isaura.git

# install zairachem
python -m pip install -e .

# create extra conda envirnoments

# install molmap
MOLMAP_ENVIRONMENT='molmap'
conda create -n $MOLMAP_ENVIRONMENT python=3.6 -y
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate $MOLMAP_ENVIRONMENT
cd zairachem/tools/molmap/bidd-molmap/
python -m pip install -r requirements.txt
cd $WORKDIR

# install mollib
cd zairachem/tools/mollib/virtual_libraries/
bash install_linux.sh
cd $WORKDIR

