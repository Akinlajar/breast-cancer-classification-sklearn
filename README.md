# Breast Cancer Classification with Scikit-Learn

A binary classification project that predicts whether a breast mass is **benign** or **malignant** using the UCI Breast Cancer Wisconsin dataset. Trains a Logistic Regression model via a scikit-learn pipeline, evaluates performance with a confusion matrix and classification report, and explores the sensitivity–specificity trade-off through ROC curve analysis.

## Dataset

**Source:** [UCI Breast Cancer Wisconsin (Original)](https://archive-beta.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+diagnostic)

699 patient records derived from digitised fine-needle aspirate (FNA) images of breast masses. Nine cell-morphology features scored on a 1–10 scale:

| # | Feature | Range |
|---|---------|-------|
| 1 | Clump Thickness | 1 – 10 |
| 2 | Uniformity of Cell Size | 1 – 10 |
| 3 | Uniformity of Cell Shape | 1 – 10 |
| 4 | Marginal Adhesion | 1 – 10 |
| 5 | Single Epithelial Cell Size | 1 – 10 |
| 6 | Bare Nuclei | 1 – 10 |
| 7 | Bland Chromatin | 1 – 10 |
| 8 | Normal Nucleoli | 1 – 10 |
| 9 | Mitoses | 1 – 10 |

**Target:** Benign (0) vs Malignant (1) — 458 benign, 241 malignant.

## Workflow

### 1. Data Loading & Exploration
- Loaded the CSV with pandas, treating `?` as NaN.
- Inspected class balance, summary statistics, and distributions.
- Generated an automated EDA report using `ydata-profiling`.

### 2. Preprocessing
- **Missing values:** Column 6 (*Bare Nuclei*) had 16 missing entries, imputed with the column mode (1.0).
- **Target encoding:** Converted the original labels (2 = benign, 4 = malignant) to binary (0 and 1).
- **Train/test split:** 70/30, stratified by class (`random_state=42`).

### 3. Model Pipeline
A scikit-learn `Pipeline` that standardises features before fitting:

```
StandardScaler → LogisticRegression
```

### 4. Evaluation

**Confusion Matrix:**

|  | Predicted Benign | Predicted Malignant |
|--|-----------------|-------------------|
| **Actual Benign** | 133 | 5 |
| **Actual Malignant** | 4 | 68 |

**Per-class metrics:**

| Metric | Benign (0) | Malignant (1) |
|--------|-----------|--------------|
| Precision | 0.97 | 0.93 |
| Recall | 0.96 | 0.94 |
| F1-Score | 0.97 | 0.94 |

**Overall accuracy: 95.7%**

**Sensitivity (recall for malignant):** 94.4% — the model correctly identifies 94.4% of malignant cases.

**Specificity (recall for benign):** 96.4% — the model correctly rules out 96.4% of benign cases.

### 5. ROC Curve & Threshold Analysis

Used `predict_proba` to obtain malignant-class probabilities and swept the decision threshold to generate an ROC curve.

**AUC Score: 0.993** — the model has excellent discriminative ability.

**Key findings from varying the threshold:**

- **Threshold = 0:** Every patient is classified as malignant → Sensitivity = 100%, Specificity = 0%. You catch all cancers but flag every healthy patient too.
- **Threshold = 1:** No patient is classified as malignant → Sensitivity = 0%, Specificity = 100%. You never raise a false alarm but miss every cancer.
- **Target sensitivity ≈ 95%:** The ROC curve reveals the corresponding specificity trade-off at that operating point.

## Tech Stack

- Python 3
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (`LogisticRegression`, `StandardScaler`, `Pipeline`, `train_test_split`, `confusion_matrix`, `classification_report`, `roc_curve`, `roc_auc_score`)
- ydata-profiling

## How to Run

```bash
git clone https://github.com/<your-username>/breast-cancer-classification-sklearn.git
cd breast-cancer-classification-sklearn

pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling jupyter

jupyter notebook
```

Open `Breast_cancer_classification.ipynb` and run all cells. Update the CSV path in the data-loading cell to match your local file location.

## Project Structure

```
breast-cancer-classification-sklearn/
├── Breast_cancer_classification.ipynb   # Main notebook
├── data/
│   └── breast-cancer-wisconsin.csv
└── README.md
```

## License

Educational project — part of a supervised learning lab module.
