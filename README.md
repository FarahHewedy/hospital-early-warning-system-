# Hospital Early-Warning & Operational Intelligence System

### Hospital Command Center — Predict, Detect, Explain

## Overview

Hospital operations are constantly affected by changes in patient demand, bed capacity, and waiting lists. Monitoring these factors across multiple specialties and locations can be challenging for hospital management teams.

This project presents an end-to-end **Hospital Early-Warning & Operational Intelligence System** designed to analyze hospital operational data, forecast potential occupancy pressure, and provide management-oriented insights to support capacity planning.

The system combines Machine Learning, data analysis, a REST API, and an interactive dashboard into one integrated prototype.

## Project Objectives

- Forecast next-quarter hospital occupancy at specialty and location level.
- Identify potential operational pressure before it escalates.
- Monitor bed capacity and historical occupancy trends.
- Analyze waiting-list indicators to provide additional operational context.
- Present insights through an interactive dashboard to support management decisions.

## Data Sources

The project uses aggregated hospital datasets published by **Public Health Scotland (PHS)**, covering:

- Hospital activity
- Hospital bed capacity
- Hospital waiting lists

The datasets are processed and prepared through the project's data analysis and preprocessing workflow.

## System Architecture

The solution consists of three main components:

1. **Data Processing & Machine Learning**
   - Data cleaning and preprocessing.
   - Exploratory Data Analysis (EDA).
   - Feature engineering.
   - Time-aware model evaluation.
   - Model persistence using Joblib.

2. **FastAPI Backend**
   - Exposes a REST API.
   - Provides a `/predict` endpoint.
   - Receives hospital operational indicators and returns an occupancy forecast.

3. **Streamlit Dashboard**
   - Displays historical bed occupancy trends.
   - Provides next-quarter occupancy forecasts through the API.
   - Highlights operational pressure levels.
   - Presents contextual operational factors.
   - Supports CSV export of forecast summaries.
   - Includes waiting-list analysis.

## Machine Learning Approach

The prediction task focuses on estimating the change in hospital occupancy for the next quarter.

The system evaluates a machine learning model against a simple persistence baseline using chronological validation.

### Model Evaluation

| Model | MAE |
|---|---:|
| Persistence Baseline | 2.362 |
| Extra Trees | 2.557 |

**Key observation:** The persistence baseline achieved a lower MAE than Extra Trees in the evaluated setup.

This highlights the importance of comparing machine learning models against simple baselines rather than assuming that a more complex model will perform better.

## Key Technical Considerations

### 1. Time-Aware Validation

Since the task involves forecasting future hospital occupancy, chronological validation was used to preserve the temporal order of observations.

### 2. Data Quality and Missing Values

Missing occupancy-related values were handled carefully. In particular, values were not blindly replaced with zero when available bed counts were zero, as this could distort operational calculations.

### 3. Dataset Granularity

Although the hospital activity dataset contains approximately 82,573 rows, individual location-specialty time series contain substantially fewer quarterly observations.

This distinction is important when designing and evaluating forecasting models.

## Technology Stack

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Backend:** FastAPI
- **Dashboard:** Streamlit
- **API Communication:** Requests
- **Data Visualization:** Streamlit charts

## Project Structure

```text
hospital-early-warning-system/
│
├── data/
│   ├── Hospital Beds.csv
│   ├── Hospital Activity.csv
│   └── Hospital Waiting.csv
│
├── artifacts/
│   ├── hospital_occupancy_pipeline.joblib
│   └── model_metadata.json
│
├── notebooks/
│   └── hospital_management.ipynb
│
├── presentation/
│   └── hospital_care_presentation.pdf
│
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

*Note: Update the filenames in this structure if your uploaded files use different names.*

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/FarahHewedy/hospital-early-warning-system.git
cd hospital-early-warning-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI backend

```bash
uvicorn main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### 4. Run the Streamlit dashboard

Open a second terminal and run:

```bash
streamlit run app.py
```

The dashboard communicates with the FastAPI backend through the `/predict` endpoint.

## Limitations & Future Improvements

- Expand the available historical time series to improve forecasting reliability.
- Investigate additional forecasting approaches and compare them against the persistence baseline.
- Integrate more complete and dynamically sourced operational indicators into prediction requests.
- Add model explainability methods such as SHAP for quantitative feature-level explanations.
- Improve automated testing and deployment configuration.

## Disclaimer

This project is an educational and portfolio-oriented operational analytics prototype. It is intended to demonstrate data science, machine learning, and software integration skills.

It is **not a clinical decision-making system** and should not be used as a substitute for professional healthcare or hospital management judgment.

## Author

**Farah Hewedy**

Computer Engineering Graduate | Aspiring Machine Learning Engineer & Data Scientist

[GitHub Profile](https://github.com/FarahHewedy)
