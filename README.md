# Automated ML for Chemistry

**WORK IN PROGRESS**

This package provides an AutoML solution to perform supervised ML on chemistry datasets.

We prioritise computational performance for low-resourced settings. Training takes time but is feasible on a conventional computer.

Works on regression and classification.

## Fit

```bash
eoschem fit --input INPUT_CSV --output MODEL_FOLDER
```

## Predict

```bash
eoschem predict --model MODEL_FOLDER --output OUTPUT_FOLDER
```
