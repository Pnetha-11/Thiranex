# Google Play Store Apps - Data Cleaning & Visualization Project

This project focuses on cleaning, analyzing, and visualizing raw Google Play Store app data. It demonstrates a complete data analytics pipeline: from downloading a messy dataset to data preprocessing, exploratory data analysis (EDA), static plotting, and building a high-fidelity interactive dashboard.

## Project Structure

```text
google_play_analysis/
│
├── data/
│   ├── googleplaystore_raw.csv        # Downloaded raw dataset (messy)
│   └── googleplaystore_cleaned.csv    # Cleaned and processed dataset
│
├── src/
│   ├── clean_data.py                  # Downloads raw data, cleans, and exports files
│   ├── eda_analysis.py                # Computes statistical summaries and correlation matrix
│   └── generate_visualizations.py     # Generates and saves 6 static matplotlib/seaborn charts
│
├── dashboard/
│   ├── index.html                     # Premium dark-theme glassmorphic dashboard
│   ├── styles.css                     # Custom responsive styles and animations
│   ├── app.js                         # Dynamic filtering and Chart.js integration
│   └── googleplaystore_cleaned.js     # Cleaned dataset exported as a JS variable (CORS-friendly)
│
├── reports/
│   ├── eda_raw_insights.txt           # Console output logs of the EDA analysis
│   ├── data_analysis_report.md        # Comprehensive Markdown report with embedded plots
│   └── images/                        # Generated static plots (.png format)
│       ├── rating_distribution.png
│       ├── category_distribution.png
│       ├── price_vs_rating.png
│       ├── installs_by_type.png
│       ├── category_vs_installs.png
│       └── correlation_heatmap.png
│
└── PROJECT_DOCUMENTATION.md           # This project documentation file
```

---

## Getting Started

### Prerequisites

You need **Python 3.8+** and standard data science libraries installed. You can install all requirements using pip:

```bash
pip install pandas numpy matplotlib seaborn
```

---

## How to Run the Data Pipeline

Open your terminal, navigate to the `src` directory, and run the scripts in sequence:

### 1. Data Cleaning & Export
Downloads the raw dataset from GitHub, cleans it (handling missing ratings, duplicates, corrupted columns, standardizing sizes to MB, and formats), and exports a cleaned CSV and JSON/JS variable.

```bash
cd google_play_analysis/src
python clean_data.py
```

### 2. Exploratory Data Analysis
Computes summary statistics, pricing metrics, category summaries, and the correlation matrix, saving them to `reports/eda_raw_insights.txt`.

```bash
python eda_analysis.py
```

### 3. Generate Static Visualizations
Generates 6 distinct analytical plots using Matplotlib/Seaborn and saves them to `reports/images/` to be displayed in the report.

```bash
python generate_visualizations.py
```

---

## Viewing the Interactive Dashboard

To satisfy premium web design guidelines, we built an interactive dashboard in the `dashboard` directory:

1.  Navigate to the `dashboard/` directory.
2.  Double-click `index.html` to open it in any modern browser (Chrome, Edge, Firefox, Safari).
3.  **No Server Required**: We exported the cleaned dataset as a Javascript variable (`googleplaystore_cleaned.js`). This avoids browser CORS security blocks that occur when loading raw JSON files via `file://` protocols, allowing the dashboard to run instantly and locally on double-click.

### Dashboard Features:
*   **KPI Cards**: Shows real-time matching App count, Average Rating, Total Installs, and Paid App ratio (with average price of paid apps) that automatically recalculate based on active filters.
*   **Search Box**: Instantly filters the apps by title.
*   **Category Filter**: Dropdown menu populated dynamically with all available play store categories.
*   **Pricing model toggle**: Toggle between Free and Paid apps.
*   **Minimum Rating Slider**: Recalculate dashboard statistics based on rating thresholds.
*   **Dynamic Chart.js Visualizations**: Updates 4 beautiful charts in real-time when filters are changed.
*   **Ranked Table**: Displays the top 10 matching apps sorted by review volume for deep-dive inspection.
*   **Premium Visuals**: Stylized with a dark neon gradient background, glassmorphic cards (`backdrop-filter` blur), fluid transitions, custom range sliders, and typography.

---

## Detailed Data Processing Details

### Raw Data Inconsistencies & Issues Handled:
1.  **Shifted Rows**: One entry ("Life Made WI-Fi Touchscreen Photo Frame") was shifted due to a missing category column, setting the rating to `19.0` and installs to `'Free'`. The script dropped this row.
2.  **Duplicate Apps**: There were 1,181 duplicate entries (same app name scraper snapshots). The cleaning script sorted entries by `Reviews` descending and retained only the most recent (highest reviews) entry.
3.  **Numeric Conversions**:
    *   `Installs` (e.g., `10,000+` -> `10000`)
    *   `Price` (e.g., `$4.99` -> `4.99`)
    *   `Reviews` (e.g., converted to integers)
4.  **Size Standardizing**: App sizes were formatted in kilobytes ('k' suffix) and megabytes ('M' suffix) or labeled `"Varies with device"`. We parsed and unified all sizes to Megabytes (MB) as float values.
5.  **Imputation of Missing Values**: Missing ratings and sizes were imputed using the median rating/size of their respective category, preventing bias and preserving data volume.
