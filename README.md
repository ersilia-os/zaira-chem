# Out-of-the-Box Supervised Machine Learning for Chemistry Datasets

**WORK IN PROGRESS**

This package provides an automated solution to perform supervised machine learning on chemistry datasets.

We prioritise computational performance for low-resourced settings. Training takes time but is feasible on a conventional computer.

Works on regression and classification. Single task or multitask.

## Fit

```bash
autosml fit --input INPUT_CSV --output MODEL_FOLDER
```

## Predict

```bash
autosml predict --model MODEL_FOLDER --output OUTPUT_FOLDER
```
