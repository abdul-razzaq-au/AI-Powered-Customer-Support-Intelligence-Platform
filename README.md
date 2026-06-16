# AI-Powered Customer Support Intelligence Platform
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)

An end-to-end Machine Learning platform that automates customer support ticket analysis using Natural Language Processing (NLP), classical Machine Learning, Explainable AI (SHAP), and API deployment.

The system predicts:
- Ticket Type from customer ticket text using DistilBERT
- Ticket Priority using tabular Machine Learning
- Expected Resolution Time using regression models

The final solution is deployed as a FastAPI application and containerized using Docker.

---

# Project Overview

Customer support teams receive thousands of tickets daily, making manual classification and prioritization inefficient.

This project builds an intelligent support analytics system that can:

- Automatically classify incoming tickets
- Predict urgency level
- Estimate resolution time
- Provide model explanations using SHAP
- Serve predictions through an API endpoint

---

# Problem Statement

Develop an AI-powered customer support system capable of analyzing incoming tickets and providing:

1. Automated ticket categorization
2. Priority prediction
3. Resolution time estimation

The objective is to reduce manual effort, improve response times, and assist support teams in handling customer issues efficiently.

---

# Solution Architecture

```
Customer Ticket Input
          |
          |
          +----------------------+
          |                      |
          v                      v
   Text Processing          Structured Features
          |                      |
          |                      |
          v                      v
    DistilBERT Model       ML Prediction Models
          |                      |
          |                      |
          v                      v
    Ticket Category       Priority + Resolution Time
          |
          |
          v
      FastAPI Backend
          |
          |
          v
      Docker Deployment
```

---

# Dataset Features

The dataset contains customer support ticket information including:

## Text Features
- Ticket Subject
- Ticket Description
- Combined Ticket Text

## Customer Features
- Customer Age
- Customer Gender

## Product Information
- Product Purchased

## Ticket Information
- Ticket Channel
- Date of Purchase
- Ticket Age

---

# Machine Learning Pipeline

## 1. Text Classification - DistilBERT

### Objective
Predict ticket category from customer text.

### Pipeline

```
Ticket Subject + Description
          |
Text Cleaning
          |
Tokenization
          |
DistilBERT Transformer
          |
Ticket Category Prediction
```

Text preprocessing included:

- Lower casing
- HTML removal
- URL removal
- Contraction expansion
- Punctuation removal
- Stopword removal
- Lemmatization

---

## 2. Ticket Priority Prediction

### Objective

Predict ticket urgency level:

- Low
- Medium
- High
- Critical


### Models Evaluated

- Decision Tree
- Random Forest
- XGBoost
- LightGBM


### Final Model

**LightGBM Classifier**

Features used:

```
Customer Age
Customer Gender Encoded
Product Purchased Encoded
Ticket Channel Encoded
Ticket Age
Character Count
Word Count
Average Word Length
```

---

## 3. Resolution Time Prediction

### Objective

Predict expected ticket resolution duration.

### Models Evaluated

- XGBoost Regression
- LightGBM Regression


### Final Model

**XGBoost Regressor**

Features used:

```
Customer Age
Customer Gender Encoded
Product Purchased Encoded
Ticket Channel Encoded
Ticket Age
Character Count
Word Count
Average Word Length
Combined Text Features (TF-IDF)
```

---

# Explainable AI

To improve model transparency, SHAP (SHapley Additive exPlanations) was implemented.

Generated explanations include:

- SHAP Summary Plots
- SHAP Waterfall Plots
- SHAP Dependence Plots

These explain:

- Feature importance
- Individual prediction contributions
- Feature impact direction

For NLP predictions:

- Token-level attention visualization was generated for DistilBERT predictions.

---

# MLflow Experiment Tracking

All machine learning experiments were tracked using MLflow.

Tracked information:

- Model parameters
- Evaluation metrics
- Model artifacts
- Experiment comparisons

---

# FastAPI Deployment

The trained models are wrapped inside a FastAPI application.

The API accepts:

```json
{
  "ticket_subject": "",
  "ticket_description": "",
  "customer_age": 0,
  "customer_gender": "",
  "product_purchased": "",
  "ticket_channel": "",
  "date_of_purchase": ""
}
```

and returns:

```json
{
  "ticket_type": "",
  "ticket_priority": "",
  "estimated_resolution_time": 0
}
```

---

# Deployment Architecture

```
FastAPI Application

        |
        |
 Docker Container

        |
        |
Cloud Deployment

(Hugging Face Spaces)

Live API:
https://armaaz-ai-powered-customer-support-intelligence-platform.hf.space/docs

---

# Project Structure

```
AI-Powered-Customer-Support-Intelligence-Platform

│
├── data/
│   └── processed/
│
├── models/
│   ├── distilbert/
│   ├── priority/
│   ├── regression/
│   └── encoders/
│
├── notebooks/
│   ├── feature_engineering.ipynb
│   ├── baseline_models.ipynb
│   ├── advanced_classical_ML.ipynb
│   ├── regression_task.ipynb
│   ├── customer_segmentation_kmeans.ipynb
│   └── explainability.ipynb
│
├── deployment/
│   ├── Dockerfile
│   └── app/
│       ├── main.py
│       ├── model_loader.py
│       ├── priority_pipeline.py
│       ├── regression_pipeline.py
│       ├── distilbert_pipeline.py
│       ├── feature_engineering.py
│       ├── text_preprocessing.py
│       ├── schemas.py
│       └── requirements.txt
│
└── README.md
```

---

# Installation

Clone repository:

```bash
git clone https://github.com/abdul-razzaq-au/AI-Powered-Customer-Support-Intelligence-Platform.git

cd AI-Powered-Customer-Support-Intelligence-Platform
```

Install dependencies:

```bash
pip install -r deployment/app/requirements.txt
```

---

# Running Locally

Navigate to:

```bash
cd deployment/app
```

Run:

```bash
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

to access Swagger API documentation.

---

# Docker Usage

Build image:

```bash
docker build -f deployment/Dockerfile -t customer-support-ai .
```

Run container:

```bash
docker run -p 8000:8000 customer-support-ai
```

---

# Business Impact

This platform can help organizations:

- Automatically route support tickets
- Prioritize urgent customer issues
- Reduce manual classification workload
- Improve SLA compliance
- Estimate workload requirements
- Provide transparent AI-driven decisions

---

# Limitations

Current limitations:

- Limited performance due to dataset complexity
- Unknown categorical values require fallback handling
- Resolution time prediction depends on available historical patterns
- Text classification performance depends on ticket quality

---

# Future Improvements

Potential enhancements:

- Replace label encoding with production-grade categorical encoders
- Add continuous model retraining pipeline
- Integrate real-time ticket ingestion
- Add monitoring using MLflow Model Registry
- Improve resolution time prediction using larger datasets
- Add chatbot-based customer interaction layer

---

# Technologies Used

## Machine Learning
- Python
- Scikit-learn
- XGBoost
- LightGBM
- PyTorch

## NLP
- Hugging Face Transformers
- DistilBERT
- TF-IDF

## Explainability
- SHAP

## Deployment
- FastAPI
- Docker
- Hugging Face Spaces / Render

## Experiment Tracking
- MLflow

---

# Author

**Abdul Razzaq**

AI-Powered Customer Support Intelligence Platform
