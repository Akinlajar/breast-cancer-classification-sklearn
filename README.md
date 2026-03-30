# Breast Cancer Classification with Scikit-Learn

A binary classification project that predicts whether a breast mass is **benign** or **malignant** using the UCI Breast Cancer Wisconsin dataset and a Logistic Regression pipeline built with scikit-learn.

## Dataset

**Source:** [UCI Breast Cancer Wisconsin (Original)](https://archive-beta.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+diagnostic)

The dataset contains 699 patient records from digitised fine-needle aspirate (FNA) images of breast masses. Nine cell-morphology features are scored on a 1–10 scale:

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

**Target:** Benign (2) vs Malignant (4) — 458 benign, 241 malignant.

## Approach

### 1. Exploratory Data Analysis
- Generated an automated profiling report using `ydata-profiling` to inspect distributions, correlations, and data quality at a glance.
- Checked class balance and summary statistics across all features.

### 2. Data Preprocessing
- **Missing values:** Column 6 (*Bare Nuclei*) contained 16 missing entries (encoded as `?` in the raw CSV). These were imputed with the column mode (1.0).
- **Feature / label split:** Columns 1–9 used as features, column 10 as the target label.
- **Train / test split:** 70 % training, 30 % test, stratified by class (`random_state=42`).

### 3. Model Pipeline
Built a scikit-learn `Pipeline` to ensure preprocessing is applied consistently:

```
StandardScaler → LogisticRegression
```

### 4. Evaluation
- Confusion matrix (visualised as a Seaborn heatmap)
- Accuracy, precision, recall, and F1-score via `classification_report`

## Results

| Metric | Benign (2) | Malignant (4) |
|--------|-----------|--------------|
| Precision | 0.97 | 0.93 |
| Recall | 0.96 | 0.94 |
| F1-Score | 0.97 | 0.94 |

**Overall accuracy: 95.7 %**

The model performs well on both classes. The dataset is reasonably balanced (65 % benign / 35 % malignant), so accuracy is a reliable metric here.

## Tech Stack

- Python 3
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (`LogisticRegression`, `StandardScaler`, `Pipeline`, `train_test_split`, `confusion_matrix`, `classification_report`)
- ydata-profiling

## How to Run

```bash
git clone https://github.com/<your-username>/breast-cancer-classification-sklearn.git
cd breast-cancer-classification-sklearn

pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling jupyter

jupyter notebook
```

Open `Day1-Block1-Exercises.ipynb` and run all cells. Update the CSV file path in the first code cell to match your local data location.

## Project Structure

```
breast-cancer-classification-sklearn/
├── Day1-Block1-Exercises.ipynb   # Main notebook
├── data/
│   └── breast-cancer-wisconsin.csv
└── README.md
```

## License

Educational project — part of a supervised learning lab module.

