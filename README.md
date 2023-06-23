[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://www.gnu.org/licenses/agpl-3.0) [![DOI](https://zenodo.org/badge/379620165.svg)](https://zenodo.org/badge/latestdoi/379620165)

[![documentation](https://img.shields.io/badge/-Documentation-purple?logo=read-the-docs&logoColor=white)](https://ersilia.gitbook.io/ersilia-book/chemistry-tools/automated-activity-prediction-models/accurate-automl-with-zairachem) [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=Python&logoColor=white)](https://github.com/psf/black)



# ZairaChem: Automated ML-based (Q)SAR

ZairaChem is the first library of Ersilia's family of tools devoted to providing **out-of-the-box** machine learning solutions for biomedical problems. In this case, we have focused on (Q)SAR models. (Q)SAR models take chemical structures as input and give as output predicted properties, typically pharmacological properties such as bioactivity against a certain target.

Both Ersilia and Zaira are cities described in Italo Calvino's book 'Invisible Cities' (1972). Ersilia is a "trading city" where inhabitants stretch strings from the corners of the houses to establish the relationships that sustain the life of the city. When the strings become too numerous, they rebuild Ersilia elsewhere, and their network of relationships remains. Zaira is a "city of memories". It contains its own past written in every corner, scratched in every pole, window and bannister.

## Installation

Clone the repository in your local system
```
git clone https://github.com/ersilia-os/zaira-chem.git
cd zaira-chem
```

From the terminal, run the installation script:
```
bash install_linux.sh
```

By default, a Conda enviroment named `zairachem` will be created. Activate it:

```
conda activate zairachem
```

## Usage

ZairaChem can be run as a command line interface. To learn more about the ZairaChem commands, see the help command_

```bash
zairachem --help
```

### Quick start

ZairaChem expects a comma- or tab-separated file containing two columns: a "smiles" column with the molecules in SMILES format and an "activity" column with the activity values. 

To get started, let's load an example classification task from [Therapeutic Data Commons](https://tdcommons.ai/). 

```bash
zairachem example --file_name input.csv
```

This file can be split into train and test sets.

```bash
zairachem split -i input.csv
```

The command above will generate two files your working directory, named train.csv and test.csv. By default, the train:test ratio is 80:20.

### Fit

You can train a model as follows:

```bash
zairachem fit -i train.csv -m model
```

This command will run the full ZairaChem pipeline and produce a model folder with processed data, model checkpoints, and reports. If no cut-off is specified for the classification, ZairaChem will establish an internal cut-off to determine Category 0 and category 1. The output results will always provide the probability of a molecule being Category 1.
Alternatively, you can set your preferred cuto-off with the following command:
```bash
zairachem fit -i train.csv -c 0.1 -d low -m model
```
Where the '-c' indicates the cut-off of the activity values and the '-d' specifies the direction. If set to 'low', values <= c will be considered 1 and if set to 'high', values => c will be considered 1.

### Predict

You can then run predictions on the test set:

```bash
zairachem predict -i test.csv -m model -o test
```

ZairaChem will run predictions using the checkpoints stored in model and store results in the test directory. Several performance plots will be generated alongside prediction outputs.

## Additional Information

For further technical details, please read the [ZairaChem page](https://ersilia.gitbook.io/ersilia-book/chemistry-tools/automated-activity-prediction-models/accurate-automl-with-zairachem) of the Ersilia gitbook, which describes each major step in the ZairaChem pipeline.

A corresponding manuscript for the ZairaChem pipeline is currently available as a [preprint](https://www.biorxiv.org/content/10.1101/2022.12.13.520154v1).

## About us

Learn about the [Ersilia Open Source Initiative](https://ersilia.io)!
