# AnalyticsVidhya's Black Friday Data Hack

Online data science hackathon held Nov 20–22, 2015 by Analytics Vidhya.
https://www.analyticsvidhya.com/datahack/contest/black-friday-data-hack

> The original link in this README (`http://datahack.analyticsvidhya.com/contest/black-friday-data-hack/`) no longer resolves directly — it now redirects to the URL above. The contest itself has concluded and the page is kept only as an archive.

## Overview

The challenge was to predict the purchase amount of products bought by customers during Black Friday, given historical transaction data including customer demographics (gender, age, occupation, city category) and product category information.

## Dataset

| File | Rows | Columns | Description |
|---|---|---|---|
| `train.csv` | 550,068 | 12 | Labeled training data |
| `test.csv` | 233,599 | 11 | Unlabeled test data (no `Purchase` column) |
| `Sample_Submission_Tm9Lura.csv` | 233,599 | 3 | Submission format template |

Columns: `User_ID`, `Product_ID`, `Gender`, `Age`, `Occupation`, `City_Category`, `Stay_In_Current_City_Years`, `Marital_Status`, `Product_Category_1`, `Product_Category_2`, `Product_Category_3`, and `Purchase` (the regression target, train only).

## Repository structure

The notebooks expect the following layout, with the notebooks themselves run from inside `notebook/`:

```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── Sample_Submission_Tm9Lura.csv
├── notebook/
│   ├── init_model.ipynb
│   └── Stacked_Model.ipynb
└── submission/      # output folder, created when the notebooks are run
```

The notebooks read data via relative paths (`../data/...`) and write predictions to `../submission/...`, so this folder structure must exist before running them — it isn't fully reflected in the current flat upload, where all files sit side by side instead.

## Notebooks

- **init_model.ipynb** — baseline approach: label-encodes the categorical features, then trains an `xgboost` regressor; also tries a TF-IDF–transformed version of the features with `xgboost`, and an `ExtraTreesRegressor` on the combined raw + TF-IDF feature set.
- **Stacked_Model.ipynb** — a two-level stacking ensemble: ten first-stage models (`xgboost`, `ExtraTreesRegressor`, `RandomForestRegressor` variants) are trained on half of the training data, their out-of-fold predictions become meta-features for a second-stage `xgboost` model trained on the other half; also includes standalone Ridge/Lasso baselines evaluated with k-fold cross-validation.

`*-checkpoint.ipynb` files are Jupyter's autosave copies of the two notebooks above and can be safely deleted or excluded from version control (e.g. via a `.gitignore` entry for `.ipynb_checkpoints/`).

## Requirements / known issues

- The notebooks are written in **Python 2** syntax (e.g. `print input.shape` instead of `print(input.shape)`) and will raise a `SyntaxError` if run as-is under Python 3.
- They import `from sklearn.cross_validation import KFold`, a module that was deprecated and then removed in scikit-learn 0.20+. On a modern install, replace this with `from sklearn.model_selection import KFold` (and adjust the `KFold(n, n_folds=5)` call signature, since the modern API takes `n_splits` and is called without the sample count as the first argument).
- Dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`.

## Running

```bash
pip install numpy pandas scikit-learn xgboost jupyter
cd notebook
jupyter notebook
```
