# Predictive Modeling Using Machine Learning

A complete Machine Learning web application that predicts outcomes based on user-provided datasets. Built with Flask, Scikit-learn, and Plotly.

## Features

- **Dataset Upload**: Upload CSV files and get instant statistical analysis.
- **Data Preprocessing**: Automatic handling of missing values, label encoding, and feature scaling.
- **Machine Learning Models**:
  - Classification: Logistic Regression, Decision Tree, Random Forest, SVM.
  - Regression: Linear Regression, Decision Tree, Random Forest, SVM.
- **Visualization Dashboard**: Interactive charts for Confusion Matrix, ROC Curve, Feature Importance, and Prediction Plots.
- **Prediction Module**: Interactive form for real-time predictions with confidence scores.
- **Responsive UI**: Modern dashboard with glassmorphism design.

## Tech Stack

- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript (ES6+), Bootstrap, Plotly.js.
- **Backend**: Python, Flask.
- **ML/DS**: Pandas, NumPy, Scikit-learn, Joblib.

## Installation & Setup

1. **Clone the project**:
   ```bash
   cd "Predictive Modeling Using Machine Learning"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the Dashboard**:
   Open `http://127.0.0.1:5000` in your browser.

## How to Use

1. **Upload**: Go to the "Dataset Upload" tab and select a CSV file (examples provided in `static/uploads/`).
2. **Preprocess**: Select the target column you want to predict and click "Preprocess".
3. **Train**: Choose an algorithm and click "Train Model".
4. **Evaluate**: View metrics and interactive charts to assess model performance.
5. **Predict**: Enter feature values in the form to get a prediction.

## Project Structure

- `app.py`: Flask backend logic and ML pipeline.
- `static/`: Contains CSS, JS, and uploaded datasets.
- `templates/`: HTML dashboard.
- `models/`: Stores the trained model files.
- `requirements.txt`: Python dependencies.
