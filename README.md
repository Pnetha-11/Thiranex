# Antigravity Prediction - Exploratory Data Analysis (EDA)

This project performs a comprehensive Exploratory Data Analysis (EDA) on an antigravity-related experimental dataset. The goal is to discover hidden patterns, clean experimental anomalies, map non-linear physical interactions (such as superconductive thermal thresholds and resonance peaks), and prepare the data for predictive machine learning models.

---

## 🚀 Quick Start

### 1. Installation of Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Generate the Dataset
Create the raw simulated lab experiment dataset (contains missing data, outliers, and physical anomalies):
```bash
python scripts/generate_data.py
```
This writes `data/raw_antigravity_data.csv` (1,000 records of experiments).

### 3. Run the EDA & Cleaning Pipeline
Execute the analysis script to clean anomalies, run statistics, extract feature importances, and output visual plots:
```bash
python scripts/run_eda.py
```
This outputs:
* Cleaned dataset: `data/cleaned_antigravity_data.csv`
* Summary stats: `reports/numerical_summary.txt`
* Saved charts: `reports/figures/` (histograms, boxplots, scatters, heatmaps, importance plots).

### 4. Explore the Interactive Notebook
To interact with the visualizations and see step-by-step documentation, start Jupyter Notebook:
```bash
jupyter notebook notebooks/antigravity_eda.ipynb
```
*(Alternatively, you can open and run this notebook in VS Code or Google Colab).*

---

## 📂 Project Structure

```
C:\Users\Pavan\Desktop\antigravity 23/
├── data/
│   ├── raw_antigravity_data.csv          # Generated raw dataset with missing values & sensor anomalies
│   └── cleaned_antigravity_data.csv      # Imputed and cleaned dataset ready for ML modeling
├── notebooks/
│   └── antigravity_eda.ipynb            # Jupyter Notebook with inline visualizations and documentation
├── scripts/
│   ├── generate_data.py                  # Code generating the synthetic experimental dataset
│   ├── run_eda.py                        # Preprocessing, analysis, and visualization pipeline
│   └── build_notebook.py                 # Helper script that built the Jupyter Notebook
├── reports/
│   ├── correlation_analysis_report.md    # Analysis of Pearson correlation vs. non-linear patterns
│   ├── final_insights_recommendations.md # Physical insights, cleaning details, and engineering guide
│   ├── numerical_summary.txt             # Raw numerical output tables and matrices
│   └── figures/                          # Saved high-resolution visualizations
│       ├── outliers_boxplot.png          # Boxplot comparing raw and cleaned variables
│       ├── temperature_distribution.png  # KDE distribution of ambient temperature
│       ├── gravity_reduction_vs_temp.png # Bivariate scatter charts detailing relationships
│       ├── correlation_heatmap.png       # Pearson correlation coefficient matrix
│       └── feature_importance.png        # Random Forest relative feature importance scores
└── README.md                             # This overview and guide
```

---

## 📊 Feature Reference

Each record represents a laboratory trial measuring gravitational field changes:

| Parameter | Unit | Description |
| :--- | :---: | :--- |
| **`experiment_id`** | Text | Unique experiment identifier (e.g., EXP-0001). |
| **`timestamp`** | Date | Time of trial registration. |
| **`material_type`** | Category | Substrate: *Superconducting Ceramic*, *Bismuth-Barium Alloy*, *Metamaterial Grid*, *Graphene-Cobalt Composite*. |
| **`superconductor_phase`**| Category | Phase: *Solid*, *Superfluid*, *Plasma*, *Bose-Einstein Condensate*, *Unknown (Sensor Error)*. |
| **`ambient_temp_k`** | Kelvin | Local temperature (Normal range: 50K–350K. Includes sub-zero Kelvin anomalies). |
| **`magnetic_field_t`** | Tesla | Applied field strength (Normal range: 0T–15T. Includes 999.0T error spikes). |
| **`excitation_freq_hz`** | Hz | Applied electromagnetic vibration (Range: 100Hz–10,000Hz. Peak resonance at 4,500Hz). |
| **`chamber_pressure_pa`** | Pascals | Vacuum chamber pressure (Range: $10^{-6}$ Pa to atmospheric $10^5$ Pa). |
| **`energy_input_mj`** | mJ | Energy input supplied to generator. |
| **`net_gravity_reduction_pct`** | % | **Target Variable** (0% to 100% reduction). |

---

## 📈 Key Findings Summary

1. **Failure of Linear Metrics**: Standard Pearson correlation shows near-zero relationship between individual variables and gravity reduction (e.g., Magnetic Field shows only $r = 0.13$). This is due to the non-linear, multiplicative physics of the system.
2. **True Feature Ranking**: A non-linear Machine Learning model (Random Forest) reveals that **Magnetic Field strength (58.2% importance)** is the dominant parameter, followed by **Energy Input (12.2%)**, **Chamber Pressure (10.8%)**, and **Frequency (10.7%)**.
3. **The Resonance Peak**: Gravity reduction peaks dramatically when the electromagnetic vibration frequency is tuned between **3,500 Hz and 5,500 Hz** (peaking around **4,500 Hz**).
4. **The Thermal Threshold**: Reaching critical lift-off (defined as $>75\%$ net gravity reduction) requires ambient temperature to fall below **93 K** in a `Superconducting Ceramic` compound to trigger the highly effective `Bose-Einstein Condensate` phase.
