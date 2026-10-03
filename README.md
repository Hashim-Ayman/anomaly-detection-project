# Credit Card Fraud Detection with Unsupervised Anomaly Detection

Detect fraudulent credit card transactions **without using labels for training**. Each transaction is compressed with a dimensionality reduction algorithm and rebuilt from the compressed form. Transactions that are hard to rebuild (high **reconstruction error**) look different from the majority, so they are flagged as likely fraud.

The labels are used only to *evaluate* how well each method catches known fraud.

## Why unsupervised?

In the real world, fraud labels are scarce (only caught fraud is ever labeled) and fraud patterns keep changing. A supervised model trained on old labels can fail to adapt to new patterns, so unsupervised anomaly detection is a common approach.

## Dataset

[Credit Card Fraud Detection (Kaggle, ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 anonymized transactions made by European cardholders in September 2013, covering about 48 hours
- 492 frauds (about 0.17%), so the data is extremely imbalanced
- `V1`-`V28`: numerical features that are the output of PCA (original features are not disclosed)
- `Time` (seconds since the first transaction) and `Amount`: the only raw features
- `Class`: 1 = fraud, 0 = genuine

The dataset is not included in this repo (it is about 150 MB). Download `creditcard.csv` from Kaggle and place it as shown below.

## Project structure

```
project/
├── Data/
│   └── creditcard.csv
├── notebooks/
│   └── anomaly_detection_fixed.ipynb
└── README.md
```

The notebook reads `../Data/creditcard.csv`, so start Jupyter from the `notebooks/` folder.

## Setup

Developed on Python 3.14 and also tested on Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install scikit-learn==1.9.1 matplotlib==3.11.2 seaborn==0.13.2 numpy pandas jupyter
```

Then run:

```bash
cd notebooks
jupyter notebook anomaly_detection_fixed.ipynb
```

Use **Run All** to regenerate every plot and table.

## What the notebook does

1. **Load and inspect** the data (`head`, `info`, `describe`).
2. **Exploratory data analysis**
   - shape, missing values, duplicates
   - class imbalance
   - `Amount` (skewed, so also shown on a log scale) and `Time` (fraud rate per hour)
   - which features separate fraud from normal (effect sizes and per-class distributions)
   - correlation matrix and correlation with `Class`
   - how often fraud has extreme values (|z| > 3) compared with normal transactions
3. **Prepare the data**: stratified train/test split (67% / 33%), then `StandardScaler` fitted on the training set only.
4. **Score anomalies**: the anomaly score is the sum of squared differences between the original and reconstructed features, min-max scaled to 0-1 (0 = normal, 1 = most anomalous).
5. **Try several reconstruction methods**

   | Method | Notes |
   |---|---|
   | PCA (30 components) | Sanity check: keeps 100% of the variance, so almost no error |
   | PCA (27 components) | Main linear baseline |
   | Sparse PCA | 27 components, `alpha=0.0001` |
   | Kernel PCA (RBF) | Trained on a 5,000-row stratified subsample because the kernel matrix is too large for all rows; its metrics are noisier and not directly comparable |
   | Gaussian Random Projection | Reconstructed with the pseudo-inverse of the random matrix |
   | Mini-batch Dictionary Learning | Non-linear; reconstructed as `codes @ dictionary` |
   | FastICA | 27 components |

6. **Evaluate** every method with the precision-recall curve (average precision) and the ROC curve (AUC), plus a scatter plot of the first two components colored by true label.
7. **Apply to the test set** for the best methods (PCA and ICA). The scaler and models are fitted on the training set only, and the test scores use the **training** min/max so both sets share the same scale.

## Evaluation notes

- With fraud this rare, accuracy is meaningless. Use average precision (area under the precision-recall curve) first and ROC AUC second.
- In the author's runs, normal PCA and ICA gave the best results.
- The number of components is a hyperparameter. Too many gives almost no reconstruction error, too few gives high error for everything. The error must be large enough on rare cases to separate them from the normal ones.

## Possible next steps

- Tune the number of components (for example with a scree plot or by comparing average precision)
- Compare against other detectors such as Isolation Forest or One-Class SVM
- Use a time-based train/test split instead of a random one, since fraud patterns drift over time
- Choose an anomaly-score threshold based on the cost of missed fraud versus false alarms

## Author

Hashim Ayman