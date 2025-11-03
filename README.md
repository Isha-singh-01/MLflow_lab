# MLflow_lab

## Overview

In this lab, I have built an end-to-end experiment tracking pipeline for text summarization using open-source LLMs integrated with MLflow.
The workflow performs multi-prompt evaluation, auto-logs results to MLflow, and stores generated summaries and metrics for later comparison.

## What This Lab Does

- Loads configuration from .env (MLflow tracking URI, Hugging Face token, etc.)
- Initializes MLflow tracking and creates a new experiment called OpenSource_GenAI_Summarization
- Loads multiple article .txt files from the /articles directory.
- For each article and for each prompt style, it:
  - Generates a summary using a local Hugging Face model (e.g., facebook/bart-large-cnn)
  - Logs parameters, metrics, and artifacts to MLflow
  - Evaluates summary quality using a mock LLM-judge scoring function
- Selects the best model based on evaluation metrics
- Registers the best model to MLflow Model Registry
- Promotes the model to Staging for deployment
- All experiment runs are versioned, reproducible, and viewable in the MLflow UI.

## Folder Structure
```
MLflow_lab/
│
├── MLflow-llmTracing.py        # Main experiment runner script
├── model_registry.py           # Model Registration script
├── requirements.txt             # All dependencies
├── .env                         # Configuration (MLflow URI, Hugging Face token)
├── articles/                    # Sample input articles (.txt)
├── screenshots/                 # Screenshots from the lab session
└── mlartifacts/                 # MLflow artifacts (auto-created)
```
