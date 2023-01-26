[![DOI](https://zenodo.org/badge/379620165.svg)](https://zenodo.org/badge/latestdoi/379620165)

# ZairaChem: Automated ML-based (Q)SAR

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

### Quick start

ZairaChem expects a comma- or tab-separated file containing molecules in SMILES format and activity values. 

To get started, let's load an example classification task from [Therapeutic Data Commons](https://tdcommons.ai/). 

```bash
zairachem example --classification --file_name input.csv
```

This file can be split into train and test sets.

```bash
zairachem split -i input.csv
```

The command above will generate two files in the current folder, named train.csv and test.csv. By default, the train:test ratio is 80:20.

### Fit

You can train a model as follows:

```bash
zairachem fit -i train.csv -m model
```

This command will run the full ZairaChem pipeline and produce a model folder with processed data, model checkpoints, and reports.

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
