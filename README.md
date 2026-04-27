# Healthcare Premium Prediction

A machine learning web application that predicts annual health insurance premiums for individuals based on personal, lifestyle, and medical attributes. Built with Python, XGBoost, and Streamlit — deployed live at [healthcarepremiumprediction.streamlit.app](https://healthcarepremiumprediction-ufs9ahmsfgntfmn6lqrkvf.streamlit.app/).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Live Demo](#live-demo)
3. [Architecture Overview](#architecture-overview)
4. [Repository Structure](#repository-structure)
5. [The Problem: Why One Model Was Not Enough](#the-problem-why-one-model-was-not-enough)
6. [Step 1 — Data Loading & Cleaning](#step-1--data-loading--cleaning)
7. [Step 2 — Exploratory Data Analysis](#step-2--exploratory-data-analysis)
8. [Step 3 — Feature Engineering](#step-3--feature-engineering)
9. [Step 4 — Error Analysis & Data Segmentation](#step-4--error-analysis--data-segmentation)
10. [Step 5 — Model Training (Segmented)](#step-5--model-training-segmented)
11. [Step 6 — Prediction Pipeline](#step-6--prediction-pipeline)
12. [Step 7 — Streamlit Application](#step-7--streamlit-application)
13. [Step 8 — Deployment on Streamlit Cloud](#step-8--deployment-on-streamlit-cloud)
14. [Tech Stack](#tech-stack)
15. [How to Run Locally](#how-to-run-locally)

---

## Project Overview

Insurance companies price premiums based on risk — but risk profiles vary dramatically across age groups. A young, healthy person under 25 has a fundamentally different cost structure compared to older adults with chronic conditions. Training a single model across this entire population leads to systematic prediction errors for one segment or the other.

This project tackles that problem head-on by:

- Training a unified model on the full dataset first
- Performing residual error analysis to identify where the model fails
- Discovering that age under 25 was the root cause of ~30% of extreme errors
- Splitting the data into two segments: **Young (age <= 25)** and **Rest (age > 25)**
- Training separate, optimized XGBoost models for each segment
- Routing predictions at inference time using a single age-based condition

The result is a production-grade prediction system with extreme errors (residuals above 10%) reduced to well below 1% of the test set.

---

## Live Demo

**App URL:** https://healthcarepremiumprediction-ufs9ahmsfgntfmn6lqrkvf.streamlit.app/

**GitHub Repo:** https://github.com/Sajalagarwal-ca/healthcare_premium_prediction

Enter any combination of age, income, medical history, lifestyle, and plan type to get an instant annual premium prediction.

---

## Architecture Overview

```
Raw Data (premiums.xlsx)
        |
        v
+-------------------+
|   EDA & Cleaning  |  --> Handle nulls, duplicates, negative values, outliers
+-------------------+
        |
        v
+----------------------+
|  Feature Engineering |  --> Risk score, encoding, VIF analysis, scaling
+----------------------+
        |
        v
+------------------------------+
|  Full Dataset Model Training |  --> Linear Regression, Ridge, XGBoost
|  + Residual Error Analysis   |  --> ~30% extreme errors traced to age <= 25
+------------------------------+
        |
   +----+----+
   |         |
   v         v
Young       Rest
(age<=25)  (age>25)
   |         |
   v         v
XGBoost   XGBoost
Model     Model
   |         |
   +----+----+
        |
        v
+------------------------+
|   prediction_helper.py |  --> Age-based routing + preprocessing + scaling
+------------------------+
        |
        v
+------------------+
|    main.py       |  --> Streamlit UI (12-field input form)
+------------------+
        |
        v
  Streamlit Cloud
  (Live Deployment)
```

At inference time, `prediction_helper.py` checks `age <= 25` and routes to the correct model and scaler pair automatically — the user never sees this complexity.

---

## Repository Structure

```
healthcare_premium_prediction/
|
|-- artifacts/                          # Serialized models and scalers
|   |-- model_young.joblib              # XGBoost model for age <= 25
|   |-- model_rest.joblib               # XGBoost model for age > 25
|   |-- scaler_young.joblib             # MinMaxScaler + columns for young segment
|   |-- scaler_rest.joblib              # MinMaxScaler + columns for rest segment
|
|-- app/                                # (Streamlit app folder for cloud deploy)
|
|-- ml_premium_prediction.ipynb         # Phase 1: Full dataset model (baseline)
|-- data_segmentation.ipynb             # Error analysis + segmentation discovery
|-- ml_premium_prediction_young.ipynb   # Phase 2: Young segment model
|-- ml_premium_prediction_rest.ipynb    # Phase 2: Rest segment model
|-- ml_premium_prediction_young_with_gr.ipynb  # With Genetical Risk feature
|-- ml_premium_prediction_rest_with_gr.ipynb   # With Genetical Risk feature
|
|-- premiums.xlsx                       # Full raw dataset
|-- premiums_young.xlsx                 # Filtered: age <= 25
|-- premiums_rest.xlsx                  # Filtered: age > 25
|-- premiums_young_with_gr.xlsx         # Young segment + genetical risk column
|
|-- prediction_helper.py                # Core prediction logic and preprocessing
|-- main.py                             # Streamlit UI application
|-- requirements.txt                    # Python dependencies
|-- README.md
```

---

## The Problem: Why One Model Was Not Enough

### Initial Observation

After training an XGBoost model on the full dataset (`premiums.xlsx`), the overall R2 score looked acceptable. However, plotting the residuals revealed a concerning pattern — a significant cluster of predictions were off by more than 10% from actual values.

```python
extreme_error_threshold = 10
extreme_results_df = results_df[np.abs(results_df['diff_pct']) > extreme_error_threshold]

# Result: ~30% of test set had extreme errors
print(extreme_results_df.shape[0] * 100 / X_test.shape[0])
```

### Root Cause: Age Segment Behaviour

Slicing the extreme error cases by age revealed that the vast majority of large residuals belonged to customers aged 25 and under. Young individuals have a fundamentally different premium structure — lower baseline risk, more sensitivity to lifestyle factors, and a compressed premium range — which a general model trained across all ages fails to capture adequately.

### The Fix: Segment and Conquer

The dataset was split at the age-25 boundary into two separate Excel files and two independent model training pipelines:

| Segment | Condition | Dataset File | Model File |
|---|---|---|---|
| Young | age <= 25 | premiums_young.xlsx | model_young.joblib |
| Rest | age > 25 | premiums_rest.xlsx | model_rest.joblib |

After segmentation, extreme errors dropped to under 0.3% of the test set for both models — a massive improvement in real-world prediction quality.

---

## Step 1 — Data Loading & Cleaning

**Notebook:** `ml_premium_prediction_rest.ipynb` (same pattern in both segment notebooks)

```python
df = pd.read_excel("premiums_rest.xlsx")

# The young model has Genetical_Risk natively; the rest model adds it as a placeholder
df['Genetical_Risk'] = 0

# Standardize column names
df.columns = df.columns.str.replace(" ", "_").str.lower()
```

### Null Values

```python
df.isna().sum()
df.dropna(inplace=True)
```

### Duplicates

```python
df.duplicated().sum()
df.drop_duplicates(inplace=True)
```

### Fixing Negative Dependants

The `number_of_dependants` column contained negative values (data entry errors). These were corrected by taking their absolute value rather than dropping the rows:

```python
df['number_of_dependants'] = df['number_of_dependants'].abs()
```

### Outlier Treatment: Age

Records with age above 100 were removed as physiologically implausible:

```python
df1 = df[df.age <= 100]
```

### Outlier Treatment: Income

A 99.9th percentile cap was applied to remove extreme income outliers while preserving valid high earners:

```python
quantile_threshold = df1.income_lakhs.quantile(0.999)
df2 = df1[df1.income_lakhs <= quantile_threshold].copy()
```

---

## Step 2 — Exploratory Data Analysis

### Numeric Feature Distributions

Histograms with KDE overlays were plotted for all numeric columns to understand skewness and value ranges:

```python
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for i, column in enumerate(numeric_columns):
    ax = axs[i // 3, i % 3]
    sns.histplot(df2[column], kde=True, ax=ax)
    ax.set_title(column)
plt.tight_layout()
plt.show()
```

### Scatter Plots vs Target

Scatter plots of each numeric feature against `annual_premium_amount` revealed which features have a linear vs. non-linear relationship with the target:

```python
numeric_features = ['age', 'income_lakhs', 'number_of_dependants', 'genetical_risk']

fig, axes = plt.subplots(1, len(numeric_features), figsize=(18, 6))
for ax, column in zip(axes, numeric_features):
    sns.scatterplot(x=df2[column], y=df2['annual_premium_amount'], ax=ax)
    ax.set_title(f'{column} vs. Annual Premium Amount')
plt.tight_layout()
plt.show()
```

### Categorical Cleaning: Smoking Status

The `smoking_status` column had three different encodings for the same "non-smoker" category. These were standardized:

```python
df2['smoking_status'].replace({
    'Not Smoking': 'No Smoking',
    'Does Not Smoke': 'No Smoking',
    'Smoking=0': 'No Smoking'
}, inplace=True)
```

---

## Step 3 — Feature Engineering

### Medical History Risk Score

Instead of one-hot encoding the `medical_history` text column (which would create sparse, hard-to-interpret features), a domain-informed risk score was calculated:

```python
risk_scores = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0
}

# Split compound conditions like "Diabetes & High blood pressure"
df2[['disease1', 'disease2']] = df2['medical_history'].str.split(" & ", expand=True).apply(
    lambda x: x.str.lower()
)
df2['disease1'] = df2['disease1'].fillna('none')
df2['disease2'] = df2['disease2'].fillna('none')

# Sum the risk scores for both diseases
df2['total_risk_score'] = 0
for disease in ['disease1', 'disease2']:
    df2['total_risk_score'] += df2[disease].map(risk_scores)

# Normalize to 0-1 range
max_score = df2['total_risk_score'].max()
min_score = df2['total_risk_score'].min()
df2['normalized_risk_score'] = (df2['total_risk_score'] - min_score) / (max_score - min_score)
```

This converts the raw text column into a single continuous feature that captures actuarial risk in a principled way, with heart disease weighted highest (8) and no disease at 0.

### Ordinal Encoding

Insurance plan and income level are ordinal in nature and were encoded with ordered integers:

```python
df2['insurance_plan'] = df2['insurance_plan'].map({'Gold': 3, 'Silver': 2, 'Bronze': 1})
df2['income_level'] = df2['income_level'].map({'<10L': 1, '10L - 25L': 2, '25L - 40L': 3, '> 40L': 4})
```

### One-Hot Encoding for Nominal Columns

Nominal categorical columns (no natural order) were one-hot encoded with `drop_first=True` to avoid the dummy variable trap:

```python
nominal_cols = ['gender', 'region', 'marital_status', 'bmi_category', 'smoking_status', 'employment_status']
df3 = pd.get_dummies(df2, columns=nominal_cols, drop_first=True, dtype=int)
```

### Multicollinearity Check with VIF

Variance Inflation Factor (VIF) was calculated to detect and remove multicollinear features before scaling:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Column'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_df

calculate_vif(X)
# income_level showed high VIF due to correlation with income_lakhs
X_reduced = X.drop('income_level', axis="columns")
```

`income_level` was dropped because it was highly correlated with `income_lakhs` — keeping both would inflate the coefficient estimates.

### Feature Scaling

A `MinMaxScaler` was applied to continuous numeric features before model training:

```python
from sklearn.preprocessing import MinMaxScaler

cols_to_scale = ['age', 'number_of_dependants', 'income_level', 'income_lakhs', 'insurance_plan', 'genetical_risk']
scaler = MinMaxScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
```

The fitted scaler was serialized alongside the column list to ensure identical transformation at inference time:

```python
from joblib import dump

scaler_with_cols = {
    'scaler': scaler,
    'cols_to_scale': cols_to_scale
}
dump(scaler_with_cols, "artifacts/scaler_rest.joblib")
```

---

## Step 4 — Error Analysis & Data Segmentation

This is the most critical insight in the project.

### Residual Analysis on the Full Model

After training the initial XGBoost model on the complete dataset, residuals were computed and their percentage error distribution was plotted:

```python
y_pred = best_model.predict(X_test)
residuals = y_pred - y_test
residuals_pct = (residuals / y_test) * 100

results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'diff': residuals,
    'diff_pct': residuals_pct
})

sns.histplot(results_df['diff_pct'], kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Diff PCT')
plt.show()
```

### Identifying Extreme Errors

Records with prediction error above 10% were flagged:

```python
extreme_error_threshold = 10
extreme_results_df = results_df[np.abs(results_df['diff_pct']) > extreme_error_threshold]
print(extreme_results_df.shape[0] * 100 / X_test.shape[0])
# Output: ~30% -- far too high for a production system
```

### Segmentation Decision

Investigating the `age` column within extreme error cases revealed that young people (age <= 25) had systematically different premium structures. The full dataset was split accordingly:

- **`premiums_young.xlsx`** — customers aged 25 and under
- **`premiums_rest.xlsx`** — customers aged 26 and above

Separate EDA, feature engineering, and model training notebooks were created for each segment (`ml_premium_prediction_young.ipynb` and `ml_premium_prediction_rest.ipynb`).

Note: The rest dataset did not originally contain a `Genetical_Risk` column. A placeholder column with value `0` was added at load time to keep the feature schema consistent between both model pipelines:

```python
# In ml_premium_prediction_rest.ipynb
df['Genetical_Risk'] = 0
```

---

## Step 5 — Model Training (Segmented)

Both segment notebooks follow the same training structure. Linear Regression and Ridge were evaluated first as baselines, then XGBoost with hyperparameter tuning was used for the final model.

### Baseline: Linear Regression

```python
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
train_score = model_lr.score(X_train, y_train)
test_score = model_lr.score(X_test, y_test)
```

Feature importance was visualized from the linear model coefficients to confirm which features had the strongest effect on premium.

### Baseline: Ridge Regression

```python
from sklearn.linear_model import Ridge

model_rg = Ridge(alpha=1)
model_rg.fit(X_train, y_train)
```

Ridge added L2 regularization as a check against overfitting on the smaller segmented datasets.

### Final Model: XGBoost with RandomizedSearchCV

```python
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

model_xgb = XGBRegressor()
param_grid = {
    'n_estimators': [20, 40, 50],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
}

random_search = RandomizedSearchCV(
    model_xgb, param_grid,
    n_iter=10, cv=3, scoring='r2',
    random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
```

### Post-Training Residual Check

The same extreme error analysis was repeated on each segment model to confirm improvement:

```python
extreme_results_df = results_df[np.abs(results_df['diff_pct']) > extreme_error_threshold]
print(extreme_results_df.shape[0] * 100 / X_test.shape[0])
# Output: ~0.3% -- acceptable for production
```

### Model Export

Both models and their scalers were saved to the `artifacts/` folder:

```python
from joblib import dump

dump(best_model, "artifacts/model_rest.joblib")
dump(scaler_with_cols, "artifacts/scaler_rest.joblib")

# Repeated for the young segment:
dump(best_model, "artifacts/model_young.joblib")
dump(scaler_with_cols, "artifacts/scaler_young.joblib")
```

---

## Step 6 — Prediction Pipeline

**File:** `prediction_helper.py`

This module handles all preprocessing and routing logic at inference time. It is cleanly separated from the UI so it can be unit-tested independently.

### Loading Models at Startup

```python
import joblib

model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")
scaler_young = joblib.load("artifacts/scaler_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")
```

### Normalized Risk Score

The medical history risk score is recalculated at prediction time using the same logic as training:

```python
def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    max_score = 14  # heart disease (8) + high blood pressure or diabetes (6)
    min_score = 0
    return (total_risk_score - min_score) / (max_score - min_score)
```

### Input Preprocessing

The raw UI input dictionary is transformed into the same one-hot encoded DataFrame schema that the models were trained on:

```python
def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
        'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest',
        'marital_status_Unmarried', 'bmi_category_Obesity', 'bmi_category_Overweight',
        'bmi_category_Underweight', 'smoking_status_Occasional', 'smoking_status_Regular',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest': df['region_Northwest'] = 1
            elif value == 'Southeast': df['region_Southeast'] = 1
            elif value == 'Southwest': df['region_Southwest'] = 1
        # ... (all other fields handled similarly)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)
    return df
```

### Age-Based Scaling Routing

Each segment was scaled independently, so the correct scaler must be applied at inference time. This is where the segmentation boundary is enforced:

```python
def handle_scaling(age, df):
    scaler_object = scaler_young if age <= 25 else scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None  # schema compatibility placeholder
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis='columns', inplace=True)

    return df
```

### Final Prediction Routing

The same age boundary drives model selection:

```python
def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])
```

A single integer (annual premium in Rupees) is returned to the UI.

---

## Step 7 — Streamlit Application

**File:** `main.py`

The UI is a clean 4-row, 3-column grid collecting all 12 input fields:

```python
import streamlit as st
from prediction_helper import predict

st.title('Health Insurance Cost Predictor')

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]: age = st.number_input('Age', min_value=18, step=1, max_value=100)
with row1[1]: number_of_dependants = st.number_input('Number of Dependants', min_value=0, step=1, max_value=20)
with row1[2]: income_lakhs = st.number_input('Income in Lakhs', step=1, min_value=0, max_value=200)

with row2[0]: genetical_risk = st.number_input('Genetical Risk', step=1, min_value=0, max_value=5)
with row2[1]: insurance_plan = st.selectbox('Insurance Plan', ['Bronze', 'Silver', 'Gold'])
with row2[2]: employment_status = st.selectbox('Employment Status', ['Salaried', 'Self-Employed', 'Freelancer', ''])

# ... (rows 3 and 4 follow the same pattern)

if st.button('Predict'):
    prediction = predict(input_dict)
    st.success(f'Predicted Health Insurance Cost: {prediction}')
```

All inputs are passed to `predict()` from `prediction_helper.py`, which handles all segmentation, preprocessing, scaling, and model selection transparently.

---

## Step 8 — Deployment on Streamlit Cloud

### Prerequisites

- A GitHub account
- A Streamlit Cloud account (free at share.streamlit.io)
- The `artifacts/` folder (with all `.joblib` files) committed to the repo

### Steps

**1. Prepare `requirements.txt`**

Ensure all dependencies are listed. The key packages are:

```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
openpyxl
```

**2. Commit everything to GitHub**

```bash
git add .
git commit -m "Add models, app, and requirements"
git push origin main
```

Make sure the `artifacts/` folder is **not** in `.gitignore` — the `.joblib` files must be present in the repo for Streamlit Cloud to load them.

**3. Deploy on Streamlit Cloud**

- Go to [share.streamlit.io](https://share.streamlit.io)
- Click **New app**
- Connect your GitHub account and select `Sajalagarwal-ca/healthcare_premium_prediction`
- Set the **Main file path** to `main.py`
- Set the **Branch** to `main`
- Click **Deploy**

Streamlit Cloud will automatically install your `requirements.txt`, start the app, and provide a public URL.

**4. Verify artifact paths**

The `joblib.load()` calls in `prediction_helper.py` use relative paths (`"artifacts/model_young.joblib"`). These resolve correctly as long as Streamlit Cloud runs from the repo root, which it does by default.

**5. Automatic Redeploy**

Any push to the `main` branch will trigger an automatic redeployment. No CI/CD pipeline setup is required.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modelling | scikit-learn (LinearRegression, Ridge, MinMaxScaler), XGBoost |
| Model Serialization | joblib |
| Statistical Analysis | statsmodels (VIF) |
| Web Application | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | Git / GitHub |

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Sajalagarwal-ca/healthcare_premium_prediction.git
cd healthcare_premium_prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run main.py
```

The app will open at `http://localhost:8501` in your browser.

To retrain the models, open and run the notebooks in this order:

1. `ml_premium_prediction.ipynb` — baseline full-dataset model
2. `data_segmentation.ipynb` — error analysis and segmentation
3. `ml_premium_prediction_young.ipynb` — young segment model
4. `ml_premium_prediction_rest.ipynb` — rest segment model

New `.joblib` files will be saved to `artifacts/` and picked up automatically by the app.

---

## Author

**Sajal Agarwal** — Senior Data Scientist / Generative AI Engineer

- Portfolio: https://sajalagarwal-ca.github.io/portfolio.github.io/index.html
- LinkedIn: https://www.linkedin.com/in/agarwalsajal/
- GitHub: https://github.com/Sajalagarwal-ca
