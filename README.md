# _ML-MODELS-FOR-HEART-DISEASE-PREDICTION
THIS PROJECT IS DONE TO KNOW  WHICH MODEL WILL PERFORM WELL FOR HEARTDISEASE PREDICTION WHITH THE DATA  SET
# Heart Disease Prediction

A small machine learning project that analyzes a heart disease dataset (`heart.csv`) and builds classification models to predict whether a patient has heart disease. The analysis is implemented in a Jupyter notebook and demonstrates data inspection, simple preprocessing, and model training/evaluation with Decision Tree, Random Forest, and K-Nearest Neighbors (KNN).

---

## Project Overview

This repository contains an end-to-end exploratory analysis and modeling pipeline for predicting heart disease from clinical features. The goal is to create baseline classifiers and report evaluation metrics so you can iterate on preprocessing, feature engineering, and model selection.

Key outcomes from the notebook:
- Dataset size: 918 rows × 12 columns
- Target: `HeartDisease` (binary; 0 = no disease, 1 = disease)
- Best baseline model (in the notebook): Random Forest
  - Random Forest accuracy on test set: 0.8804
  - Decision Tree accuracy on test set: 0.8043
  - KNN accuracy on test set: 0.7011

---

## Dataset

Source file: `heart.csv` (example path used during development on Windows: `C:\Users\jesti\Downloads\heart.csv`)

Columns (as used in the notebook):
- Age — patient age in years
- Sex — M/F (encoded in preprocessing)
- ChestPainType — categorical (e.g., ATA, NAP, ASY, TA)
- RestingBP — resting blood pressure (mm Hg)
- Cholesterol — serum cholesterol in mg/dl
- FastingBS — fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- RestingECG — resting electrocardiographic results (categorical)
- MaxHR — maximum heart rate achieved
- ExerciseAngina — exercise-induced angina (Y/N)
- Oldpeak — ST depression induced by exercise relative to rest
- ST_Slope — slope of the peak exercise ST segment (categorical)
- HeartDisease — target (0 = no, 1 = yes)

Quick dataset summary from the notebook:
- Rows: 918
- Some columns contain zero values (e.g., `Cholesterol`, `RestingBP`) which may indicate missing data or outliers and should be handled before production use.
- Target distribution: ~55% positive (mean ≈ 0.553)

---

## Preprocessing (notebook)

The notebook performs the following preprocessing steps as a minimal baseline:

1. Read CSV:
   - `pd.read_csv(r"C:\Users\jesti\Downloads\heart.csv")`

2. Basic inspection:
   - `.describe()` and quick `.info()` / `.head()` checks.

3. Missing-value awareness:
   - The notebook displays data but does not perform full imputation; there are zero values in some numeric columns that likely represent missing values.

4. Label encoding of categorical variables:
   - `LabelEncoder()` used to convert:
     - `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope` → numeric codes

5. Split:
   - `train_test_split(..., test_size=0.2, random_state=42)` (80% train / 20% test)

Notes / caveats:
- Numeric zero values in `Cholesterol` and `RestingBP` likely indicate missing or invalid measurements. Consider replacing zeros with NaN and imputing (median or KNN imputer) before training.
- No scaling was applied. Some models (e.g., KNN) often benefit from feature scaling (StandardScaler or MinMaxScaler).
- Label encoding assigns integer codes but does not create one-hot encodings. For tree-based models this is fine; for linear models or distance-based methods you may prefer one-hot encoding.

---

## Models & Results (from notebook)

Three classifiers were trained and evaluated on the held-out test set. Metrics below are the notebook results (test set):

1. Decision Tree
   - Accuracy: 0.8043
   - Confusion matrix:
     [[65, 12],
      [24, 83]]

2. Random Forest (best in notebook)
   - Accuracy: 0.8804
   - Confusion matrix:
     [[66, 11],
      [11, 96]]

3. K-Nearest Neighbors (k=5)
   - Accuracy: 0.7011
   - Confusion matrix:
     [[55, 22],
      [33, 74]]

Evaluation metrics printed: accuracy, precision, recall, F1-score, confusion matrix, and classification report (per-class metrics).

---

## How to reproduce

1. Clone this repository (or place the notebook and `heart.csv` in a directory).

2. Create & activate a Python environment (recommended):
   - Using conda:
     - conda create -n heart-py python=3.9 -y
     - conda activate heart-py
   - Or using venv:
     - python -m venv venv
     - windows: venv\Scripts\activate
     - mac/linux: source venv/bin/activate

3. Install requirements:
   - pip install -r requirements.txt
   - If there's no requirements file, install:
     - pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyterlab

4. Open the notebook:
   - jupyter lab
   - or: jupyter notebook Heart\ Disease\ Prediction.ipynb (use the actual filename)

5. Load the CSV:
   - Edit the path in the notebook (if needed) to point to your `heart.csv`.
   - Example:
     - data = pd.read_csv(r"C:\Users\jesti\Downloads\heart.csv")
     - Or use a relative path: `data = pd.read_csv("data/heart.csv")`

6. Re-run cells or execute the notebook from top to bottom to reproduce results.

Optional: Run a script entrypoint (if provided) that executes preprocessing and prints metrics.

---

## Recommended improvements / Next steps

To strengthen the analysis and move toward production-ready models, consider:

- Data cleaning
  - Investigate zero values in `Cholesterol`, `RestingBP` and decide if they represent missing data; convert to NaN and impute (median or iterative/KNN imputation).
  - Check for other outliers and inconsistent categories.

- Feature engineering
  - Create derived variables (e.g., age bins, pulse recovery metrics).
  - Consider domain features or interactions.

- Proper encoding & scaling
  - Use OneHotEncoder for categorical features if using linear/logistic models.
  - Scale features (StandardScaler) for distance-based models or regularized models.

- Model evaluation
  - Use cross-validation (stratified k-fold) for more reliable estimates.
  - Tune hyperparameters (GridSearchCV or RandomizedSearchCV).
  - Evaluate calibration (calibration curve) for probability predictions.

- Class imbalance
  - If target imbalance becomes an issue, consider class weights, SMOTE, or other resampling.

- Explainability
  - Use feature importances, Permutation Importance, or SHAP values to explain model predictions.

- Packaging & deployment
  - Save the best model using `joblib` or `pickle`.
  - Add a small Flask/FastAPI wrapper or Streamlit UI for demoing predictions.

---

## Repository structure (suggested)

- README.md                # This file
- heart.csv                # Data (not always committed; add to .gitignore if needed)
- notebook.ipynb           # Jupyter notebook with code and results
- requirements.txt         # Python dependencies
- src/
  - preprocess.py
  - train.py
  - evaluate.py
- models/
  - best_model.joblib

---

## Requirements

Minimum Python packages used in the notebook:
- python >= 3.8
- pandas
- numpy
- scikit-learn
- matplotlib (optional for plots)
- seaborn (optional for plots)
- joblib (optional for saving models)
- jupyterlab / notebook (to run the notebook)

Example pip install:
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyterlab
```

---

## Reproducibility notes

- Random seeds:
  - The notebook uses `random_state=42` in `train_test_split` and the RandomForestClassifier to make results reproducible. For full reproducibility set seeds for all libraries used (numpy, sklearn, etc.) and avoid any nondeterministic operations or record their seeds.
- Environment:

