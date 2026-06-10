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

def run_project_pipeline():
    # Setup directories
    os.makedirs(os.path.join("data"), exist_ok=True)
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    os.makedirs(os.path.join("dashboard"), exist_ok=True)
    
    # Load dataset
    raw_path = os.path.join("data", "raw_churn_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Please run generate_data.py first.")
        
    df_raw = pd.read_csv(raw_path)
    df_clean = df_raw.copy()
    
    print(f"Loaded raw data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    
    # --- DATA CLEANING & PREPROCESSING ---
    print("Starting data cleaning pipeline...")
    
    # 1. Deduplication
    initial_rows = df_clean.shape[0]
    df_clean.drop_duplicates(inplace=True)
    duplicate_count = initial_rows - df_clean.shape[0]
    
    # 2. Text standardization (Standardize casing)
    df_clean['gender'] = df_clean['gender'].str.capitalize()
    df_clean['partner'] = df_clean['partner'].str.capitalize()
    
    for col in ['online_security', 'tech_support']:
        # Capitalize values except 'No internet service' which we standardize
        df_clean[col] = df_clean[col].apply(lambda x: 'No internet service' if str(x).lower() == 'no internet service' else str(x).capitalize())
        
    # 3. Handle Tenure outliers (Replace negative values with median of their contract type group)
    neg_tenures = (df_clean['tenure'] < 0).sum()
    if neg_tenures > 0:
        tenure_medians = df_clean[df_clean['tenure'] >= 0].groupby('contract')['tenure'].transform('median')
        # Mask negative values and fill with medians
        df_clean.loc[df_clean['tenure'] < 0, 'tenure'] = df_clean.loc[df_clean['tenure'] < 0, 'contract'].map(
            df_clean[df_clean['tenure'] >= 0].groupby('contract')['tenure'].median()
        )
        
    # 4. Handle Monthly Charges outliers (Negative values & 999.0 T spikes -> replace with median of internet_service group)
    outlier_charges_mask = (df_clean['monthly_charges'] < 0) | (df_clean['monthly_charges'] > 200)
    outlier_charges_count = outlier_charges_mask.sum()
    if outlier_charges_count > 0:
        charge_medians = df_clean[~outlier_charges_mask].groupby('internet_service')['monthly_charges'].median()
        df_clean.loc[outlier_charges_mask, 'monthly_charges'] = df_clean.loc[outlier_charges_mask, 'internet_service'].map(charge_medians)
        
    # 5. Clean Total Charges (Convert spaces and NaNs to floats, impute based on tenure * monthly_charges)
    # Convert empty spaces to NaN
    total_charges_raw = df_clean['total_charges'].astype(str).str.strip()
    df_clean['total_charges'] = pd.to_numeric(total_charges_raw, errors='coerce')
    
    missing_total_charges = df_clean['total_charges'].isnull().sum()
    if missing_total_charges > 0:
        # Impute missing charges with tenure * monthly_charges (logical baseline)
        null_idx = df_clean['total_charges'].isnull()
        df_clean.loc[null_idx, 'total_charges'] = df_clean.loc[null_idx, 'tenure'] * df_clean.loc[null_idx, 'monthly_charges']
        
    # Save cleaned dataset
    cleaned_path = os.path.join("data", "cleaned_churn_data.csv")
    df_clean.to_csv(cleaned_path, index=False)
    print("Data cleaning complete!")
    print(f"  - Removed duplicates: {duplicate_count} records")
    print(f"  - Corrected negative tenures: {neg_tenures} records")
    print(f"  - Corrected monthly charge outliers (<0 or 999): {outlier_charges_count} records")
    print(f"  - Imputed missing total charges: {missing_total_charges} records")
    print(f"Cleaned dataset saved to {cleaned_path}\n")
    
    # --- WRITE DATA QUALITY REPORT ---
    with open(os.path.join("reports", "data_quality_report.txt"), "w") as f:
        f.write("========================================================================\n")
        f.write("                  CUSTOMER CHURN DATA QUALITY REPORT                    \n")
        f.write("========================================================================\n\n")
        f.write(f"Raw Records: {initial_rows}\n")
        f.write(f"Cleaned Records: {df_clean.shape[0]}\n\n")
        f.write("--- ISSUES RESOLVED ---\n")
        f.write(f"1. Duplicate rows dropped: {duplicate_count}\n")
        f.write(f"2. Casing inconsistencies normalized in features: gender, partner, online_security, tech_support.\n")
        f.write(f"3. Negative tenure records resolved: {neg_tenures} records (imputed with contract group medians).\n")
        f.write(f"4. Monthly charge outliers (< $0 or = $999) resolved: {outlier_charges_count} records (imputed with internet_service group medians).\n")
        f.write(f"5. Total charges missing values resolved: {missing_total_charges} records (imputed using: tenure * monthly_charges).\n")
        
    # --- STATISTICAL ANALYSIS ---
    print("Calculating descriptive statistics...")
    num_cols = ['tenure', 'monthly_charges', 'total_charges']
    
    # Advanced descriptive stats
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
    
    # Categorical summary
    churn_counts = df_clean['churn'].value_counts()
    churn_pcts = df_clean['churn'].value_counts(normalize=True) * 100
    churn_summary = pd.DataFrame({'Count': churn_counts, 'Percentage (%)': churn_pcts})
    
    # --- HYPOTHESIS TESTING (SIGNIFICANCE) ---
    print("Performing hypothesis testing...")
    
    # 1. Chi-Square Test: Contract type vs Churn
    contingency_table = pd.crosstab(df_clean['contract'], df_clean['churn'])
    chi2, chi2_p, dof, expected = chi2_contingency(contingency_table)
    
    # 2. Welch's t-test: Tenure vs Churn
    churn_yes_tenure = df_clean[df_clean['churn'] == 'Yes']['tenure']
    churn_no_tenure = df_clean[df_clean['churn'] == 'No']['tenure']
    t_stat_tenure, p_val_tenure = ttest_ind(churn_yes_tenure, churn_no_tenure, equal_var=False)
    
    # 3. Welch's t-test: Monthly Charges vs Churn
    churn_yes_charges = df_clean[df_clean['churn'] == 'Yes']['monthly_charges']
    churn_no_charges = df_clean[df_clean['churn'] == 'No']['monthly_charges']
    t_stat_charges, p_val_charges = ttest_ind(churn_yes_charges, churn_no_charges, equal_var=False)
    
    # --- MODELING & EVALUATION ---
    print("Preparing data for machine learning models...")
    
    # Prepare features
    # Target: churn (Yes=1, No=0)
    y = df_clean['churn'].map({'Yes': 1, 'No': 0})
    X = df_clean.drop(columns=['customer_id', 'churn'])
    
    categorical_features = ['gender', 'partner', 'dependents', 'phone_service', 
                            'multiple_lines', 'internet_service', 'online_security', 
                            'online_backup', 'device_protection', 'tech_support', 
                            'streaming_tv', 'streaming_movies', 'contract', 
                            'paperless_billing', 'payment_method']
    numerical_features = ['tenure', 'monthly_charges', 'total_charges']
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])
    
    # Define models
    log_reg = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42, max_iter=500))])
                              
    rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(random_state=42))])
                             
    xgb_clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])
                              
    # Train models
    print("Training models...")
    log_reg.fit(X_train, y_train)
    xgb_clf.fit(X_train, y_train)
    
    # Hyperparameter tuning on Random Forest using GridSearchCV
    print("Running GridSearchCV for Random Forest tuning...")
    rf_param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(rf_clf, rf_param_grid, cv=3, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"  - Best Random Forest parameters: {rf_grid.best_params_}")
    
    # Model evaluation helper
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        return acc, prec, rec, f1, auc, cm, y_prob
        
    # Evaluate all models
    acc_lr, prec_lr, rec_lr, f1_lr, auc_lr, cm_lr, y_prob_lr = evaluate_model(log_reg, X_test, y_test)
    acc_rf, prec_rf, rec_rf, f1_rf, auc_rf, cm_rf, y_prob_rf = evaluate_model(best_rf, X_test, y_test)
    acc_xgb, prec_xgb, rec_xgb, f1_xgb, auc_xgb, cm_xgb, y_prob_xgb = evaluate_model(xgb_clf, X_test, y_test)
    
    # Construct model comparison dataframe
    model_comparison = pd.DataFrame({
        'Logistic Regression': [acc_lr, prec_lr, rec_lr, f1_lr, auc_lr],
        'Random Forest': [acc_rf, prec_rf, rec_rf, f1_rf, auc_rf],
        'XGBoost': [acc_xgb, prec_xgb, rec_xgb, f1_xgb, auc_xgb]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']).T
    
    print("\n=== MODEL COMPARISON TABLE ===")
    print(model_comparison)
    
    # Determine best model based on F1-Score (balanced measure)
    best_model_name = model_comparison['F1-Score'].idxmax()
    print(f"\nBest model selected (F1-score): {best_model_name}")
    
    if best_model_name == 'Logistic Regression':
        best_model = log_reg
    elif best_model_name == 'Random Forest':
        best_model = best_rf
    else:
        best_model = xgb_clf
        
    # --- VISUALIZATION GENERATION ---
    print("Generating figures...")
    sns.set_theme(style="whitegrid")
    
    # 1. Churn count plot
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df_clean, x='churn', palette='Set2', hue='churn', legend=False)
    plt.title("Distribution of Customer Churn Status", fontsize=13, weight='bold')
    plt.xlabel("Churned")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "churn_distribution.png"), dpi=200)
    plt.close()
    
    # 2. Tenure distribution by Churn status
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df_clean, x='tenure', hue='churn', fill=True, common_norm=False, palette='crest', alpha=0.4)
    plt.title("Customer Loyalty Profile: Tenure Distribution by Churn Status", fontsize=13, weight='bold')
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "tenure_distribution.png"), dpi=200)
    plt.close()
    
    # 3. Contract Churn Comparison
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_clean, x='contract', hue='churn', palette='viridis')
    plt.title("Contract Risks: Churn Status across Contract Tiers", fontsize=13, weight='bold')
    plt.xlabel("Contract Type")
    plt.ylabel("Customer Count")
    plt.legend(title="Churned")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "contract_churn_comparison.png"), dpi=200)
    plt.close()
    
    # 4. Correlation Heatmap (Numeric Columns)
    plt.figure(figsize=(8, 6))
    corr_mat = df_clean[num_cols].corr()
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title("Pearson Correlation Heatmap (Cleaned Numerical Features)", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "correlation_heatmap.png"), dpi=200)
    plt.close()
    
    # 5. Feature Importance (Random Forest Gini Importances)
    # Extract features from preprocessor
    ohe_cats = best_rf.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_cats)
    importances = best_rf.named_steps['classifier'].feature_importances_
    
    feat_imp = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    # clean labels
    clean_labels = [name.replace("contract_", "Contract: ").replace("internet_service_", "Internet: ").replace("payment_method_", "Payment: ") for name in feat_imp['Feature']]
    feat_imp['CleanFeature'] = clean_labels
    
    sns.barplot(data=feat_imp.head(10), x='Importance', y='CleanFeature', palette='magma', hue='CleanFeature', legend=False)
    plt.title("Top 10 Feature Importances (Random Forest)", fontsize=13, weight='bold')
    plt.xlabel("Gini Importance Score")
    plt.ylabel("Operational Parameters")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "feature_importance.png"), dpi=200)
    plt.close()
    
    # 6. ROC Curves comparison
    plt.figure(figsize=(8, 6))
    for model_name, y_prob, auc_score in [
        ('Logistic Regression', y_prob_lr, auc_lr),
        ('Random Forest', y_prob_rf, auc_rf),
        ('XGBoost', y_prob_xgb, auc_xgb)
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})", linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight='bold')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "roc_curves.png"), dpi=200)
    plt.close()
    
    # 7. Confusion Matrices Side-by-Side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    matrices = [('Logistic Regression', cm_lr), ('Random Forest', cm_rf), ('XGBoost', cm_xgb)]
    for i, (name, cm) in enumerate(matrices):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i], annot_kws={'size': 14})
        axes[i].set_title(f"{name} Confusion Matrix", fontsize=12, weight='bold')
        axes[i].set_xlabel("Predicted Class")
        axes[i].set_ylabel("True Class")
        axes[i].set_xticklabels(['No Churn', 'Churn'])
        axes[i].set_yticklabels(['No Churn', 'Churn'])
        
    plt.suptitle("Model Prediction Confusion Matrices", fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "confusion_matrices.png"), dpi=200)
    plt.close()
    
    # 8. Monthly Charges vs Tenure Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_clean, x='tenure', y='monthly_charges', hue='churn', style='churn', palette='coolwarm', alpha=0.6)
    plt.title("Monthly Charges vs Customer Tenure: Churn Overlay", fontsize=13, weight='bold')
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Monthly Charges ($)")
    plt.legend(title="Churned")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "monthly_charges_vs_tenure.png"), dpi=200)
    plt.close()
    
    # --- WRITE STATISTICAL SUMMARY TEXT FILE ---
    print("Writing numerical summary report...")
    with open(os.path.join("reports", "numerical_summary.txt"), "w") as f:
        f.write("========================================================================\n")
        f.write("                CUSTOMER CHURN STATISTICAL ANALYSIS REPORT               \n")
        f.write("========================================================================\n\n")
        
        f.write("=== CLEANED NUMERICAL FEATURE DESCRIPTIONS ===\n")
        f.write(desc_df.to_string())
        f.write("\n\n=== CHURN DISTRIBUTIONS ===\n")
        f.write(churn_summary.to_string())
        
        f.write("\n\n=== HYPOTHESIS TESTING RESULTS ===\n")
        f.write(f"1. Chi-Square Test for Contract Type vs Churn:\n")
        f.write(f"   Chi2-statistic = {chi2:.4f}, p-value = {chi2_p:.4e}\n")
        f.write(f"   Interpretation: {'Statistically Significant. Contract choice strongly relates to churn.' if chi2_p < 0.05 else 'Not significant.'}\n\n")
        
        f.write(f"2. Welch's t-test comparing Customer Tenure (Churn vs No Churn):\n")
        f.write(f"   t-statistic = {t_stat_tenure:.4f}, p-value = {p_val_tenure:.4e}\n")
        f.write(f"   Interpretation: {'Statistically Significant. Deflected customers have significantly shorter lifespans.' if p_val_tenure < 0.05 else 'Not significant.'}\n\n")
        
        f.write(f"3. Welch's t-test comparing Monthly Charges (Churn vs No Churn):\n")
        f.write(f"   t-statistic = {t_stat_charges:.4f}, p-value = {p_val_charges:.4e}\n")
        f.write(f"   Interpretation: {'Statistically Significant. Churned customers have significantly higher monthly bills.' if p_val_charges < 0.05 else 'Not significant.'}\n\n")
        
        f.write("=== MACHINE LEARNING MODEL COMPARISON ===\n")
        f.write(model_comparison.to_string())
        f.write(f"\n\nBest model selected: {best_model_name}\n")
        
        f.write("\n=== FEATURE IMPORTANCE RANKING (RANDOM FOREST) ===\n")
        f.write(feat_imp.head(15).to_string(index=False))
        
    # --- MODEL COEFFICIENTS EXPORT FOR JAVASCRIPT CALCULATOR ---
    print("Extracting Logistic Regression parameters for JS calculator...")
    # To run a logistic regression model in Javascript, we need the fitted coefficients and the standard scaler means/stds.
    # Let's rebuild a Logistic Regression on standard variables to make it clean to write in JS.
    # We will use the trained Logistic Regression model coefficients from the pipeline.
    # The pipeline preprocessor is: ColumnTransformer
    # We extract numeric means/stds
    scaler_means = log_reg.named_steps['preprocessor'].named_transformers_['num'].mean_
    scaler_stds = log_reg.named_steps['preprocessor'].named_transformers_['num'].scale_
    
    # We extract OHE categories
    ohe_encoder = log_reg.named_steps['preprocessor'].named_transformers_['cat']
    ohe_categories = []
    for col, categories in zip(categorical_features, ohe_encoder.categories_):
        for cat in categories:
            ohe_categories.append(f"{col}_{cat}")
            
    all_lr_features = numerical_features + ohe_categories
    lr_coefs = log_reg.named_steps['classifier'].coef_[0]
    lr_intercept = log_reg.named_steps['classifier'].intercept_[0]
    
    coef_dict = {}
    for feat, coef in zip(all_lr_features, lr_coefs):
        coef_dict[feat] = coef
        
    js_model_metadata = {
        'intercept': lr_intercept,
        'coefficients': coef_dict,
        'numerical_stats': {
            'tenure': {'mean': scaler_means[0], 'std': scaler_stds[0]},
            'monthly_charges': {'mean': scaler_means[1], 'std': scaler_stds[1]},
            'total_charges': {'mean': scaler_means[2], 'std': scaler_stds[2]}
        }
    }
    
    # --- EXPORT DATASET AS JS FOR CORS-FREE DASHBOARD ---
    print("Exporting data and ML metadata for dashboard...")
    df_js = df_clean.copy()
    
    # Generate predictions and probabilities using the best model for dashboard charts
    best_probs = best_model.predict_proba(df_js.drop(columns=['customer_id', 'churn']))[:, 1]
    best_preds = np.where(best_probs >= 0.5, 'Yes', 'No')
    
    df_js['churn_prob'] = best_probs
    df_js['churn_pred'] = best_preds
    
    records = df_js.to_dict(orient='records')
    
    js_file_path = os.path.join("dashboard", "data_source.js")
    with open(js_file_path, "w", encoding="utf-8") as f:
        f.write("// Telemetry and ML prediction data - Customer Churn Prediction\n")
        f.write(f"const CHURN_METADATA = {json.dumps(js_model_metadata, indent=2)};\n\n")
        f.write(f"const CHURN_DATA = {json.dumps(records, indent=2)};\n")
        
    print(f"Data exported to JavaScript: {js_file_path}")
    print("Project pipeline completed successfully! Outputs generated in reports/ and dashboard/")

if __name__ == "__main__":
    run_project_pipeline()
