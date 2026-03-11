---
title: Air Pollution Dashboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
---


# AutoViz-AI-Universal-Data-Visualization-Dashboard


# 📈 Automatic Data Visualization Dashboard

A dark-themed, business intelligence style web application built with **Python** and **Streamlit** that automatically generates a full interactive analytics dashboard from any uploaded CSV dataset — no configuration required.

---

## ✨ Features

### Upload & Load
- Upload any CSV file via the sidebar
- Automatically parses and loads the dataset with Streamlit caching for performance
- Bundled sample dataset available for instant preview

### Smart Column Detection
Automatically detects and classifies every column into:
- **Numeric** — integers and floats
- **Categorical** — text and object columns
- **Datetime** — columns whose values can be parsed as dates

### Dashboard Sections

| Section | What is shown |
|---|---|
| **Dataset Overview** | Preview table, schema, data types, missing value counts, column type chips |
| **KPI Metrics** | Total rows, total columns, numeric features, categorical features |
| **Automatic Chart Generation** | Histogram · Scatter · Pie · Bar · Box · Line in a 3-column grid |
| **Distribution Analysis** | Secondary histogram, KDE density plot |
| **Category Analysis** | Frequency bar chart, count plot |
| **Relationship Analysis** | Cross-feature scatter plots, trend comparison line charts |
| **Correlation Analysis** | Seaborn correlation heatmap for all numeric features |
| **Missing Value Analysis** | Bar chart of missing counts per column, row-level missing heatmap |
| **Dataset Insights** | Auto-generated summary, highest correlated features, highest variance columns |
| **Interactive Visualization Builder** | Custom chart from sidebar — X axis, Y axis, and chart type selection |
| **Detailed Statistics** | Full `describe()` output including variance |

### Chart Types Available
- Histogram
- Box plot
- Scatter plot
- Line chart
- Bar chart
- Pie / Donut chart
- KDE (Kernel Density Estimate) plot
- Correlation heatmap (Seaborn)
- Count plot (Seaborn)

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework and UI layout |
| [Pandas](https://pandas.pydata.org) | Data loading and transformation |
| [NumPy](https://numpy.org) | Numerical operations |
| [Plotly Express](https://plotly.com/python/plotly-express/) | Interactive charts |
| [Seaborn](https://seaborn.pydata.org) | Heatmaps, KDE plots, count plots |
| [Matplotlib](https://matplotlib.org) | Figure rendering for Seaborn charts |

---

## 🚀 Getting Started

### 1. Clone or download the project

```bash
cd "DashBoard project"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
DashBoard project/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── global air pollution dataset.csv  # Bundled sample dataset
└── README.md                       # This file
```

---

## 🧩 Code Structure

Key reusable functions inside `app.py`:

| Function | Description |
|---|---|
| `load_data()` | Cached CSV loader using `pd.read_csv` |
| `detect_column_types()` | Auto-classifies columns as numeric, categorical, or datetime |
| `build_summary()` | Extended `describe()` output with variance column |
| `generate_histograms()` | Plotly histograms for numeric columns |
| `generate_bar_charts()` | Plotly bar charts for categorical columns |
| `generate_pie_charts()` | Plotly donut charts for categorical columns |
| `generate_scatter_plots()` | Plotly scatter plots for top-variance numeric pairs |
| `generate_heatmap()` | Seaborn correlation heatmap |
| `generate_kde_plots()` | Seaborn KDE density plots |
| `generate_count_plots()` | Seaborn count plots |
| `generate_box_plots()` | Plotly box plots for outlier analysis |
| `generate_line_charts()` | Plotly line charts using datetime or index |
| `generate_missing_values_chart()` | Plotly bar chart of missing value counts |
| `generate_missing_heatmap()` | Seaborn row-level missing data heatmap |
| `generate_dataset_insights()` | Top correlations, top variance, summary text |
| `render_custom_chart()` | Builds any chart type from sidebar user selection |
| `sample_dataframe()` | Downsamples large datasets for performance |

---

## 📊 Example Datasets

The dashboard works automatically with any CSV. It has been tested with:

- **Global Air Pollution Dataset** *(bundled)*
- Sales datasets
- Student performance datasets
- Healthcare / patient datasets
- Financial / stock datasets

---

## ⚙️ Sidebar Controls

| Control | Function |
|---|---|
| CSV Upload | Load any dataset file |
| Use bundled sample | Load the built-in air pollution dataset |
| Numeric columns | Multiselect filter for numeric features |
| Categorical columns | Multiselect filter for categorical features |
| Datetime columns | Multiselect filter for datetime features |
| Chart type | Select chart type for the custom builder |
| X-axis column | Select X axis for the custom builder |
| Y-axis column | Select Y axis for the custom builder (optional) |

---

## 🖼 UI Design

- **Dark theme** with deep navy backgrounds and glassmorphism-style card panels
- **Colorful Plotly charts** using the `plotly_dark` template with custom accent palettes
- **Responsive grid layout** using Streamlit columns — charts appear in rows of 2 or 3
- **KPI metric cards** with gradient accent bars
- **Schema chips** showing detected column type labels
- **Section titles** separating each analytics area clearly

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
seaborn
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```
