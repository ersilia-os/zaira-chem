# ZairaChem: Automated ML-based (Q)SAR

**THIS REPOSITORY IS WORK IN PROGRESS**

ZairaChem is the first library of Ersilia's family of tools devoted to providing **out-of-the-box** machine learning solutions for biomedical problems. In this case, we have focused on (Q)SAR models. (Q)SAR models take chemical structures as input and give as output predicted properties, typically pharmacological properties such as bioactivity against a certain target.

## Installation

From the terminal, run the installation script:
```
bash install_linux.sh
```

By default, a Conda enviroment named `zairachem` will be created. Activate it:

```
conda activate zairachem
```

## Usage

ZairaChem can be run as a command line interface.

```bash
zairachem --help
```

### Get your data

ZairaChem expect a comma- or tab-separated file containing molecules in SMILES format and activity values. If you don't have a dataset at hand, you can create an example with the following command. 

```bash
zairachem example -f input.csv
```

### Fit

```bash
zairachem fit -i input.csv -o model_folder
```

### Predict

```bash
zairachem predict -i input.csv -m model_folder -o output_folder
```

## About us

Learn about the [Ersilia Open Source Initiative](https://ersilia.io)!