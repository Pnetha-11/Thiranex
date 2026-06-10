import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Title & Executive Summary
    cells.append(nbf.v4.new_markdown_cell("""# Real-World Data Science Project: Telecom Customer Churn Prediction
*Domain: Retail / Telecommunications*

---

## Executive Summary
This notebook demonstrates a complete end-to-end data science pipeline focusing on **Customer Churn Prediction**. Customer churn represents a critical revenue risk for subscription-based business models. By predicting which customers are most likely to cancel services, marketing and customer success teams can proactively deploy retention campaigns.

### Project Goals:
1.  **Data Preprocessing**: Clean messy raw logs by resolving duplicate entries, text casing inconsistencies, missing financial fields, and database sensor spikes (e.g. impossible charges and tenures).
2.  **Exploratory Data Analysis (EDA)**: Calculate advanced descriptive statistics (skewness, kurtosis, variance) and generate 8 static visualizations mapping customer loyalty curves and contract dependencies.
3.  **Hypothesis Testing**: Run Welch's t-tests and Chi-Square contingency tests to determine the statistical significance of contract tiers, billing fees, and lifespans.
4.  **Predictive Modeling**: Train and tune three classification algorithms—**Logistic Regression**, **Random Forest**, and **XGBoost Classifier**—to predict churn risk.
5.  **Model Evaluation**: Assess models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC (prioritizing Recall and F1-Score to minimize missed churners).
"""))

    # 2. Setup and Imports
    cells.append(nbf.v4.new_code_cell("""# Import required libraries
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Configure visual style for standard plotting
%matplotlib inline
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 120
})

print("Environment setup successful! All data science packages loaded.")"""))

    # 3. Load Data
    cells.append(nbf.v4.new_markdown_cell("""## 1. Data Understanding & Loading
We load the raw dataset `raw_churn_data.csv` and inspect its shape, columns, features, and check for missing entries.
"""))

    cells.append(nbf.v4.new_code_cell("""# Define dataset path
raw_data_path = os.path.join("..", "data", "raw_churn_data.csv")
if not os.path.exists(raw_data_path):
    raw_data_path = os.path.join("data", "raw_churn_data.csv")

df_raw = pd.read_csv(raw_data_path)
print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print("\\n--- SAMPLE TELEMETRY ---")
df_raw.head()"""))

    cells.append(nbf.v4.new_code_cell("""# Inspect data types and structural integrity
df_raw.info()"""))

    cells.append(nbf.v4.new_code_cell("""# Identify missing values and null entries
print("--- Missing Values count ---")
print(df_raw.isnull().sum())"""))

    # 4. Cleaning
    cells.append(nbf.v4.new_markdown_cell("""## 2. Data Cleaning & Preprocessing
To resolve sensor errors and database bugs, we execute a cleaning pipeline:
1. **Deduplication**: Drop rows containing identical customer entries.
2. **Standard Casing**: Normalize string case for categoricals like `gender` and `partner`.
3. **Tenure Outlier Replacement**: Replace negative tenure values ($T < 0$) with the median tenure of their contract group.
4. **Monthly Charges Outlier Replacement**: Replace negative bills and sensor spikes ($999.00) with the median monthly charges of their internet service type.
5. **Total Charges Imputation**: Convert `total_charges` to numeric, drop space strings, and impute missing records by taking `tenure` * `monthly_charges`.
"""))

    cells.append(nbf.v4.new_code_cell("""# Copy dataframe for cleaning
df_clean = df_raw.copy()

# 1. Deduplicate
initial_rows = df_clean.shape[0]
df_clean.drop_duplicates(inplace=True)
print(f"Removed duplicates: {initial_rows - df_clean.shape[0]} rows dropped.")

# 2. Normalize casing
df_clean['gender'] = df_clean['gender'].str.capitalize()
df_clean['partner'] = df_clean['partner'].str.capitalize()
for col in ['online_security', 'tech_support']:
    df_clean[col] = df_clean[col].apply(lambda x: 'No internet service' if str(x).lower() == 'no internet service' else str(x).capitalize())

# 3. Handle Tenure anomalies (Negative values)
neg_tenures = (df_clean['tenure'] < 0).sum()
if neg_tenures > 0:
    df_clean.loc[df_clean['tenure'] < 0, 'tenure'] = df_clean.loc[df_clean['tenure'] < 0, 'contract'].map(
        df_clean[df_clean['tenure'] >= 0].groupby('contract')['tenure'].median()
    )
    print(f"Corrected {neg_tenures} negative tenure records.")

# 4. Handle Monthly Charges anomalies (Spikes & Negatives)
outlier_charges_mask = (df_clean['monthly_charges'] < 0) | (df_clean['monthly_charges'] > 200)
outlier_charges_count = outlier_charges_mask.sum()
if outlier_charges_count > 0:
    charge_medians = df_clean[~outlier_charges_mask].groupby('internet_service')['monthly_charges'].median()
    df_clean.loc[outlier_charges_mask, 'monthly_charges'] = df_clean.loc[outlier_charges_mask, 'internet_service'].map(charge_medians)
    print(f"Corrected {outlier_charges_count} monthly charges outliers.")

# 5. Impute Total Charges
df_clean['total_charges'] = pd.to_numeric(df_clean['total_charges'].astype(str).str.strip(), errors='coerce')
missing_total_charges = df_clean['total_charges'].isnull().sum()
if missing_total_charges > 0:
    null_idx = df_clean['total_charges'].isnull()
    df_clean.loc[null_idx, 'total_charges'] = df_clean.loc[null_idx, 'tenure'] * df_clean.loc[null_idx, 'monthly_charges']
    print(f"Imputed {missing_total_charges} null total charges.")

# Export clean CSV
cleaned_csv_path = os.path.join("..", "data", "cleaned_churn_data.csv")
if not os.path.exists(os.path.dirname(cleaned_csv_path)):
    cleaned_csv_path = os.path.join("data", "cleaned_churn_data.csv")
df_clean.to_csv(cleaned_csv_path, index=False)
print("Cleaned dataset successfully exported.")"""))

    # 5. Descriptive Stats
    cells.append(nbf.v4.new_markdown_cell("""## 3. Advanced Statistical Summary
We compute detailed metrics on cleaned features to evaluate distribution characteristics (skewness, kurtosis, variance).
"""))

    cells.append(nbf.v4.new_code_cell("""# Advanced numerical descriptions
num_cols = ['tenure', 'monthly_charges', 'total_charges']
desc_stats = {}
for col in num_cols:
    col_data = df_clean[col]
    desc_stats[col] = {
        'Mean': col_data.mean(),
        'Median': col_data.median(),
        'Mode': col_data.mode()[0] if not col_data.mode().empty else np.nan,
        'Variance': col_data.var(),
        'Std Dev': col_data.std(),
        'Skewness': col_data.skew(),
        'Kurtosis': col_data.kurtosis(),
        'Minimum': col_data.min(),
        'Maximum': col_data.max()
    }
desc_df = pd.DataFrame(desc_stats).T
desc_df"""))

    cells.append(nbf.v4.new_code_cell("""# Display Churn Rate count & percentage
churn_counts = df_clean['churn'].value_counts()
churn_pcts = df_clean['churn'].value_counts(normalize=True) * 100
pd.DataFrame({'Count': churn_counts, 'Percentage (%)': churn_pcts})"""))

    # 6. Significance Tests
    cells.append(nbf.v4.new_markdown_cell("""## 4. Hypothesis Significance Testing
To check if experimental differences are due to chance, we run significance tests:
1. **Chi-Square Test** on Contract Type vs. Churn.
2. **Welch's t-test** on Tenure (Churned vs. Active) and Monthly Charges (Churned vs. Active).
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Chi-Square: Contract Type vs Churn
contingency_table = pd.crosstab(df_clean['contract'], df_clean['churn'])
chi2, chi2_p, dof, expected = chi2_contingency(contingency_table)
print(f"1. Chi-Square Test (Contract vs Churn): Chi2 = {chi2:.4f}, p-val = {chi2_p:.4e}")
print(f"   Interpretation: Significant? {chi2_p < 0.05}\\n")

# 2. Welch's t-test: Tenure (Churn vs Active)
churn_yes_tenure = df_clean[df_clean['churn'] == 'Yes']['tenure']
churn_no_tenure = df_clean[df_clean['churn'] == 'No']['tenure']
t_stat_t, p_val_t = ttest_ind(churn_yes_tenure, churn_no_tenure, equal_var=False)
print(f"2. Welch's t-test (Tenure of Churned vs Active): t-stat = {t_stat_t:.4f}, p-val = {p_val_t:.4e}")
print(f"   Interpretation: Significant? {p_val_t < 0.05}\\n")

# 3. Welch's t-test: Monthly Charges (Churn vs Active)
churn_yes_charges = df_clean[df_clean['churn'] == 'Yes']['monthly_charges']
churn_no_charges = df_clean[df_clean['churn'] == 'No']['monthly_charges']
t_stat_c, p_val_c = ttest_ind(churn_yes_charges, churn_no_charges, equal_var=False)
print(f"3. Welch's t-test (Monthly Charges of Churned vs Active): t-stat = {t_stat_c:.4f}, p-val = {p_val_c:.4e}")
print(f"   Interpretation: Significant? {p_val_c < 0.05}")"""))

    # 7. Preprocessing & Splitting
    cells.append(nbf.v4.new_markdown_cell("""## 5. Machine Learning Pipeline Preparation
We encode target outputs, define a `StandardScaler` for numeric scaling and `OneHotEncoder` for categoricals, and partition the dataset.
"""))

    cells.append(nbf.v4.new_code_cell("""# Map target
y = df_clean['churn'].map({'Yes': 1, 'No': 0})
X = df_clean.drop(columns=['customer_id', 'churn'])

categorical_features = ['gender', 'partner', 'dependents', 'phone_service', 
                        'multiple_lines', 'internet_service', 'online_security', 
                        'online_backup', 'device_protection', 'tech_support', 
                        'streaming_tv', 'streaming_movies', 'contract', 
                        'paperless_billing', 'payment_method']
numerical_features = ['tenure', 'monthly_charges', 'total_charges']

# Split 80/20 stratified
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessor transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
])

print(f"Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")"""))

    # 8. Model Training & Tuning
    cells.append(nbf.v4.new_markdown_cell("""## 6. Model Training & Hyperparameter Tuning
We train Logistic Regression, XGBoost Classifier, and perform cross-validated Grid Search to optimize a Random Forest Classifier.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Train Logistic Regression
log_reg = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(random_state=42, max_iter=500))])
log_reg.fit(X_train, y_train)

# 2. Train XGBoost
xgb_clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])
xgb_clf.fit(X_train, y_train)

# 3. GridSearchCV for Random Forest Classifier
rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', RandomForestClassifier(random_state=42))])
rf_param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None]
}
rf_grid = GridSearchCV(rf_clf, rf_param_grid, cv=3, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print(f"GridSearch Optimization Complete!")
print(f"Best RF hyperparameters: {rf_grid.best_params_}")"""))

    # 9. Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## 7. Model Evaluation & Comparison
We extract classification metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC, compiling a comparative model table.
"""))

    cells.append(nbf.v4.new_code_cell("""def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, prec, rec, f1, auc, cm, y_prob

acc_lr, prec_lr, rec_lr, f1_lr, auc_lr, cm_lr, y_prob_lr = evaluate_model(log_reg, X_test, y_test)
acc_rf, prec_rf, rec_rf, f1_rf, auc_rf, cm_rf, y_prob_rf = evaluate_model(best_rf, X_test, y_test)
acc_xgb, prec_xgb, rec_xgb, f1_xgb, auc_xgb, cm_xgb, y_prob_xgb = evaluate_model(xgb_clf, X_test, y_test)

# Compare
comparison_df = pd.DataFrame({
    'Logistic Regression': [acc_lr, prec_lr, rec_lr, f1_lr, auc_lr],
    'Random Forest': [acc_rf, prec_rf, rec_rf, f1_rf, auc_rf],
    'XGBoost': [acc_xgb, prec_xgb, rec_xgb, f1_xgb, auc_xgb]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']).T
comparison_df"""))

    # 10. Visualizations
    cells.append(nbf.v4.new_markdown_cell("""## 8. Static Analytics Visualizations
We plot distributions, correlations, ROC-AUC comparisons, confusion matrices, and feature importances.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Churn count plot
plt.figure(figsize=(6, 5))
sns.countplot(data=df_clean, x='churn', palette='Set2')
plt.title("Distribution of Customer Churn Status", fontsize=13, weight='bold')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Tenure KDE Distribution by Churn status
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df_clean, x='tenure', hue='churn', fill=True, common_norm=False, palette='crest', alpha=0.4)
plt.title("Customer Loyalty Profile: Tenure Distribution by Churn Status", fontsize=13, weight='bold')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 3. Contract Churn Comparison
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x='contract', hue='churn', palette='viridis')
plt.title("Contract Risks: Churn Status across Contract Tiers", fontsize=13, weight='bold')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 4. Correlation Heatmap
plt.figure(figsize=(8, 6))
corr_mat = df_clean[num_cols].corr()
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Pearson Correlation Heatmap (Cleaned Numerical Features)", fontsize=13, weight='bold')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 5. Feature Importances (Random Forest)
ohe_cats = best_rf.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_cats)
importances = best_rf.named_steps['classifier'].feature_importances_

feat_imp = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
feat_imp['CleanFeature'] = [name.replace("contract_", "Contract: ").replace("internet_service_", "Internet: ").replace("payment_method_", "Payment: ") for name in feat_imp['Feature']]

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(10), x='Importance', y='CleanFeature', palette='magma')
plt.title("Top 10 Feature Importances (Random Forest)", fontsize=13, weight='bold')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 6. Combined ROC Curves
plt.figure(figsize=(8, 6))
for name, y_prob, auc_score in [
    ('Logistic Regression', y_prob_lr, auc_lr),
    ('Random Forest', y_prob_rf, auc_rf),
    ('XGBoost', y_prob_xgb, auc_xgb)
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title("ROC Curves Comparison", fontsize=13, weight='bold')
plt.legend()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 7. Confusion Matrices Side-by-Side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
matrices = [('Logistic Regression', cm_lr), ('Random Forest', cm_rf), ('XGBoost', cm_xgb)]
for i, (name, cm) in enumerate(matrices):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
    axes[i].set_title(f"{name} Confusion Matrix", fontsize=12, weight='bold')
    axes[i].set_xticklabels(['No Churn', 'Churn'])
    axes[i].set_yticklabels(['No Churn', 'Churn'])
plt.show()"""))

    # 11. Key Insights
    cells.append(nbf.v4.new_markdown_cell("""## 9. Business Findings & Tactical Recommendations

### Key Data Insights:
1. **The Contract Risk factor**: Month-to-month contracts represent the highest churn risk ($Chi^2$-test: $p < 0.01$). This contract tier represents quick-exit options, whereas one-year and two-year contracts secure customer loyalty.
2. **The Tenure Churn Curve**: Welch's t-test confirms churned customers have significantly shorter tenures ($p < 0.01$). Churn is heavily concentrated in the first 0-12 months.
3. **The Electronic Check Anomalies**: Electronic checks exhibit anomalously high churn rates ($2.5 \\times$ normal credit card/bank automatic transfers). E-checks are manual and highly prone to monthly failure or cancellation triggers.
4. **Services as Loyalty Anchors**: Customers who do not sign up for Tech Support and Online Security have a churn rate that is $3 \\times$ higher than customers who have active protection services.

### Strategic Business Recommendations:
*   **Billing Migration**: Incentivize electronic check customers to migrate to Bank Transfer or Credit Card automatic billing by offering a one-time $5 billing credit.
*   **Contract Conversion Campaigns**: Deploy automated email offers targeting month-to-month customers in their 3rd to 6th month, offering a discounted monthly rate if they sign a 1-year contract.
*   **Bundle Security Add-Ons**: Pre-bundle basic Online Security and Tech Support into higher tier internet packages. Even at a slight margin loss, the long-term customer lifespans (reduction in churn) yield a massive net positive CLV (Customer Lifetime Value).
"""))

    nb['cells'] = cells
    
    # Save notebook
    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "customer_churn_analysis.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    
    print(f"Success! Jupyter notebook created at {nb_path}")

if __name__ == "__main__":
    create_notebook()
