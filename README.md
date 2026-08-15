# Predictive Maintenance Data Pipeline & Dashboard 🚀

An end-to-end, production-grade data engineering pipeline and interactive dashboard that predicts turbofan engine failures using the NASA C-MAPSS dataset. 

This project demonstrates scalable data processing, data quality enforcement, database modeling, and interactive visualization—the complete lifecycle of a modern data product.

## 📖 Business Value

Unplanned equipment failure is one of the most expensive problems in industrial operations. Turbofan engines that fail mid-flight or without warning cost significantly more in emergency repairs and operational downtime than engines flagged early for preventative maintenance.

This pipeline solves that problem by:
- **Ingesting raw telemetry** from 21 different sensors per engine.
- **Aggregating and smoothing** the signal to filter out high-frequency noise.
- **Flagging anomalies** when sensors deviate beyond mathematically established fleet baselines.
- **Predicting Remaining Useful Life (RUL)** by analyzing the slope of degradation.
- **Surfacing critical engines** in a low-latency Streamlit dashboard for maintenance teams.

## 🏗️ Architecture

The pipeline follows a Medallion architecture (Bronze ➔ Silver ➔ Gold), orchestrated via Python, utilizing Polars for high-performance transformations, and PostgreSQL for analytical serving.

```mermaid
graph TD
    A[NASA C-MAPSS<br>Raw Text Files] -->|Ingest| B[(Bronze Layer<br>Raw Parquet)]
    B -->|Clean & Validate| C[(Silver Layer<br>Cleaned Parquet)]
    C -->|Aggregate & Smooth| D[(Gold Layer<br>Feature Parquet)]
    D -->|Upsert via SQLAlchemy| E[(PostgreSQL<br>Data Warehouse)]
    E -->|SQL Views & Ranking| F[Streamlit<br>Dashboard]
```

### Tech Stack
* **Data Processing**: `Polars`, `Pandas`
* **Storage**: Local Parquet (Data Lake simulation)
* **Data Warehouse**: `PostgreSQL 17`
* **Application / Viz**: `Streamlit`, `Plotly`, `SQLAlchemy`
* **CI/CD / Quality**: `pytest`, `flake8`, `black`

## 📊 Features & Analytics

1. **Fleet Health Overview**: High-level KPIs showing total active engines and average remaining useful life.
2. **Interactive Deep Dive**: Users can select any engine in the fleet and plot specific raw sensor signals overlaid with 7-cycle rolling averages and confidence bands.
3. **Sensor Signal Independence**: A correlation matrix proving feature selection and highlighting multicollinearity.
4. **Data Quality Guarantees**: A live footer validating the percentage of anomalous records dropped versus total rows processed, guaranteeing dashboard trust.

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- PostgreSQL 17 (Running locally or via Docker)

### 2. Setup Environment
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive-maintenance
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Database
Copy the environment template and update it with your PostgreSQL credentials:
```bash
cp .env.example .env
# Edit .env and set DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

### 4. Run the Pipeline
Run the end-to-end orchestration script. This will provision the database, download the NASA dataset, run the Bronze ➔ Silver ➔ Gold transformations, and load the data into Postgres.
```bash
python orchestrate/run_pipeline.py
```

### 5. Launch the Dashboard
Start the Streamlit application to visualize the results:
```bash
streamlit run dashboard/app.py
```

## 🧪 Testing and CI/CD

This repository enforces data engineering best practices. The transformations and ingest logic are covered by unit tests.
To run the test suite locally:
```bash
pytest
```
Code quality is enforced via standard tools:
```bash
black .
flake8 .
```
