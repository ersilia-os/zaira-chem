# ZairaChem: out-of-the-box QSAR

**WORK IN PROGRESS**

ZairaChem is the first library of Ersilia's family of tools devoted to providing **out-of-the-box** machine learning solutions for biomedical problems. In this case, we have focused on QSAR models. QSAR models have chemical structures as input and as output they have predicted properties, typically pharmacological properties such as bioactivity against a certain target.

## Installation

### Create a conda environment

```bash
conda create -n zairachem python=3.7
conda activate zairachem
```

### Install ersilia

```bash
git clone git@github.com:ersilia-os/ersilia.git
cd ersilia
python -m pip install -e .
```

### Install isaura

```bash
git clone git@github.com:ersilia-os/isaura.git
cd isaura
python -m pip install -e .
```

### Install zairachem

```bash
git clone git@github.com:ersilia-os/zaira-chem.git
cd zaira-chem
python -m pip install -e .
```

## Usage

ZairaChem works as a command line.

```bash
zairachem --help
```

### Fit

```bash
zairachem fit -i input.csv
```
