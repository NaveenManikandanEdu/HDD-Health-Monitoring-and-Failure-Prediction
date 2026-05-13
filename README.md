# HDDPred

Enterprise-grade HDD Health Monitoring and Failure Prediction Platform powered by Machine Learning, Real-Time Telemetry, and Predictive Analytics.

---

## Overview

HDDPred is an intelligent predictive maintenance platform designed to monitor large-scale HDD infrastructures and proactively forecast potential drive failures before they occur.

The platform combines:

* Real-time telemetry monitoring
* SMART attribute analysis
* Incremental machine learning pipelines
* Automated orchestration workflows
* Enterprise dashboard visualization
* Synthetic simulation environments
* Continuous model lifecycle management

The system is built for industrial and enterprise environments where large-scale storage reliability is mission-critical.

---

# Core Features

## Real-Time HDD Monitoring

* Live telemetry ingestion
* SMART metric tracking
* Warehouse-scale monitoring
* Continuous drive health evaluation
* Multi-vault operational visibility

## Predictive Failure Detection

* Machine learning-based risk forecasting
* Short-term failure probability estimation
* Z-calibrated risk scoring
* Failure pattern clustering
* Health categorization and diagnostics

## Incremental Machine Learning Pipeline

* Automated preprocessing
* Feature engineering engine
* Incremental LightGBM training
* Continuous model retraining
* Model promotion and rollback system

## Enterprise Controller System

* Automated orchestration workflows
* Live monitoring loops
* Batch preprocessing pipelines
* Startup reconciliation
* Fault tolerance and recovery
* Model lifecycle management

## Reporting and Alerting

* PDF intelligence reports
* Email alert system
* Risk dashboards
* Historical analysis
* Operational insights

## Simulation Environment

* Synthetic HDD telemetry generation
* Warehouse simulation engine
* Failure injection testing
* Load and stress testing
* Real-time telemetry replay

---

# System Architecture

```text
HDDs / Sensors
       ↓
Data Collection Layer
       ↓
Preprocessing Engine
       ↓
Feature Engineering
       ↓
Machine Learning Layer
       ↓
Risk Evaluation & Inference
       ↓
Alerting & Reporting
       ↓
Web Dashboard / APIs
       ↓
Continuous Training Pipeline
```

---

# Repository Structure

```text
hddpred/
│
├── controller/
│   ├── core/
│   ├── emailer/
│   ├── reports/
│   └── storage/
│
├── ml/
│   ├── preprocessing/
│   ├── training/
│   ├── utils/
│   ├── checkpoints/
│   └── models/
│
├── simulator/
│   ├── data/
│   ├── raw_snapshots/
│   └── warehouse_simulator.py
│
├── web_app/
│   ├── backend/
│   ├── frontend/
│   └── config/
│
├── data/
├── docs/
├── tests/
└── scripts/
```

---

# Technology Stack

## Backend

* Python
* FastAPI
* SQLite
* JWT Authentication
* WebSockets

## Machine Learning

* LightGBM
* Scikit-learn
* Pandas
* NumPy
* Joblib

## Frontend

* Streamlit
* Plotly
* Bootstrap

## Infrastructure

* Watchdog Monitoring
* Incremental Training Pipelines
* Automated Controller System
* Parallel Processing

---

# Machine Learning Pipeline

## Data Collection

The system collects:

* SMART attributes
* HDD telemetry
* Historical failure records
* Live operational metrics

## Feature Engineering

Generated features include:

* Rolling averages
* Delta metrics
* Trend statistics
* Temporal degradation patterns
* Failure indicators

## Risk Prediction

The platform uses:

* Incremental LightGBM models
* Pattern similarity analysis
* Z-score calibration
* Failure probability estimation

## Continuous Learning

The system supports:

* Incremental retraining
* Active model promotion
* Backup and rollback mechanisms
* Automated evaluation cycles

---

# Enterprise Controller Workflow

The Controller acts as the orchestration backbone of the system.

## Responsibilities

* Monitor incoming telemetry
* Trigger preprocessing pipelines
* Manage prediction workflows
* Dispatch alerts
* Coordinate model training
* Promote validated models
* Handle fault recovery

## Model Lifecycle

```text
Archive → Validation → Training → Evaluation → Promotion → Active
```

---

# Dashboard Features

## Live Operations Center

* Fleet-wide HDD monitoring
* Warehouse telemetry visualization
* Critical risk tracking
* Real-time status monitoring

## Risk Intelligence

* Failure mechanism clustering
* Top critical drive analysis
* Health distribution visualization
* Prediction history

## Operational Monitoring

* Event logs
* Alert management
* Live telemetry feeds
* Maintenance insights

---

# PDF Intelligence Reporting

The platform automatically generates:

* Executive summaries
* Fleet health reports
* Failure mechanism charts
* Risk distribution analysis
* Top critical drive inventories

---

# Simulation Engine

The Simulation Module enables:

* Synthetic HDD telemetry generation
* Rare failure scenario testing
* Performance benchmarking
* Alert stress testing
* Pipeline validation

---

# Installation

## Clone Repository

```bash
git clone https://github.com/NaveenManikandanEdu/HDD-Health-Monitoring-and-Failure-Prediction.git
cd HDD-Health-Monitoring-and-Failure-Prediction
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the System

## Start Backend API

```bash
uvicorn web_app.backend.main:app --reload
```

## Start Streamlit Dashboard

```bash
streamlit run web_app/frontend/app.py
```

## Start Controller

```bash
python controller/main.py
```

## Start Simulator

```bash
python simulator/warehouse_simulator.py
```

---

# Future Improvements

Planned enhancements include:

* Transformer-based failure prediction
* Edge AI deployment
* Kubernetes orchestration
* Distributed streaming pipelines
* Cloud-native scaling
* Explainable AI (XAI)
* Self-healing maintenance automation

---

# Security

The system supports:

* JWT authentication
* Role-based access control
* Encrypted communications
* Audit logging
* Secure telemetry handling

---

# Screenshots

Add screenshots inside:

```text
docs/screenshots/
```

Recommended images:

* Dashboard Home
* Warning Grid
* Login Page
* PDF Reports
* Live Telemetry Monitor
* Simulator Output
* System Architecture

---

# Research Foundation

This project was developed as part of the Bachelor of Computer Applications program under the title:

> Intelligent HDD Health Forecasting System

The system focuses on predictive maintenance using machine learning and real-time telemetry analytics for enterprise HDD infrastructures.

---

# License

This project is licensed under the MIT License.

---

# Contact

For research collaboration, improvements, or enterprise discussions:

* GitHub Issues
* Pull Requests
* Project Discussions

---

# HDDPred

Transforming HDD maintenance from reactive monitoring into intelligent predictive infrastructure.
