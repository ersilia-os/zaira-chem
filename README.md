# ZairaChem: Automated ML-based (Q)SAR

ZairaChem is the first library of Ersilia's family of tools devoted to providing **out-of-the-box** machine learning solutions for biomedical problems. In this case, we have focused on (Q)SAR models. (Q)SAR models take chemical structures as input and give as output predicted properties, typically pharmacological properties such as bioactivity against a certain target.

## Installation

From the terminal, run the installation script:
```
bash install_linux.sh
```

By default, a Conda enviroment names `zairachem` will be created. Activate it:

```
conda activate zairachem
```

## Usage

ZairaChem works as a command line.

```bash
zairachem --help
```

### Fit

```bash
zairachem fit -i input.csv -o model_folder
```

### Predict

```bash
zairachem predict -i input.csv -m model_folder -o output_folder
```