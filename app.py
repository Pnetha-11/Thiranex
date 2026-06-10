import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, auc, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MODEL_FOLDER'] = 'models'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Global variables to store session data (In production, use a database or session)
data_store = {
    'df': None,
    'target': None,
    'features': [],
    'model': None,
    'task_type': None, # 'classification' or 'regression'
    'label_encoders': {},
    'scaler': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'current_data.csv')
        file.save(filename)
        
        df = pd.read_csv(filename)
        data_store['df'] = df
        
        # Basic stats
        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'null_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'summary': df.describe().to_dict()
        }
        
        preview = df.head(10).to_dict(orient='records')
        
        return jsonify({
            'message': 'File uploaded successfully',
            'stats': stats,
            'preview': preview
        })
    return jsonify({'error': 'Invalid file type. Please upload a CSV.'}), 400

@app.route('/preprocess', methods=['POST'])
def preprocess():
    if data_store['df'] is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    df = data_store['df'].copy()
    params = request.json
    target_col = params.get('target')
    impute_mode = params.get('impute_mode', 'mean') # mean, median, drop
    
    if not target_col or target_col not in df.columns:
        return jsonify({'error': 'Invalid target column'}), 400

    # 1. Handle Missing Values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                if impute_mode == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif impute_mode == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df.dropna(subset=[col], inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # 2. Encode Categorical Features
    le_dict = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    data_store['df_processed'] = df
    data_store['target'] = target_col
    data_store['label_encoders'] = le_dict
    
    # Determine task type
    if df[target_col].nunique() < 20: # Heuristic for classification
        data_store['task_type'] = 'classification'
    else:
        data_store['task_type'] = 'regression'
        
    return jsonify({
        'message': 'Preprocessing complete',
        'task_type': data_store['task_type'],
        'columns': df.columns.tolist()
    })

@app.route('/train', methods=['POST'])
def train_model():
    if 'df_processed' not in data_store:
        return jsonify({'error': 'Preprocess data first'}), 400
    
    df = data_store['df_processed']
    target = data_store['target']
    params = request.json
    model_name = params.get('model_name')
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    data_store['scaler'] = scaler
    data_store['features'] = X.columns.tolist()

    # Model selection
    task = data_store['task_type']
    if task == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC(probability=True)
        }
    else:
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'SVM': SVR()
        }
    
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 400
    
    model = models[model_name]
    model.fit(X_train_scaled, y_train)
    data_store['model'] = model
    
    y_pred = model.predict(X_test_scaled)
    
    metrics = {}
    plots = {}
    
    if task == 'classification':
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                            title="Confusion Matrix")
        plots['confusion_matrix'] = fig_cm.to_json()
        
        # ROC Curve (for binary classification mostly)
        if len(np.unique(y)) == 2:
            y_score = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={auc(fpr, tpr):.2f})",
                               labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            plots['roc_curve'] = fig_roc.to_json()
            
    else:
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2 Score': r2_score(y_test, y_pred)
        }
        # Prediction vs Actual Plot
        fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
                               title="Actual vs Predicted")
        fig_pred.add_shape(type='line', line=dict(dash='dash'), 
                           x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max())
        plots['prediction_plot'] = fig_pred.to_json()

    # Feature Importance (for some models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        plots['feature_importance'] = fig_feat.to_json()
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        if task == 'classification' and len(coefs.shape) > 1:
            coefs = np.mean(np.abs(coefs), axis=0)
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': coefs.flatten()}).sort_values(by='Importance', ascending=False)
        fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Coefficients")
        plots['feature_importance'] = fig_feat.to_json()

    # Save model
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'trained_model.joblib')
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'features': X.columns.tolist(),
        'le': data_store['label_encoders'],
        'task': task
    }, model_path)
    
    return jsonify({
        'message': f'{model_name} trained successfully',
        'metrics': metrics,
        'plots': plots
    })

@app.route('/predict', methods=['POST'])
def predict():
    model_data = joblib.load(os.path.join(app.config['MODEL_FOLDER'], 'trained_model.joblib'))
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    le_dict = model_data['le']
    task = model_data['task']
    
    input_data = request.json
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input (same as training)
    for col, le in le_dict.items():
        if col in input_df.columns:
            # Handle unknown labels if necessary
            input_df[col] = le.transform(input_df[col].astype(str))
            
    input_scaled = scaler.transform(input_df[features])
    prediction = model.predict(input_scaled)
    
    result = {
        'prediction': float(prediction[0]),
        'task': task
    }
    
    if task == 'classification' and hasattr(model, 'predict_proba'):
        prob = model.predict_proba(input_scaled)
        result['confidence'] = float(np.max(prob))
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
