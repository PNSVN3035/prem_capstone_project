# Zepto Data & AI Platform — Capstone Project

## Overview

This capstone project contains three connected modules that cover data engineering, analytics and machine learning, and a GenAI-based support assistant.

The project is maintained in one GitHub repository, with each module stored in its own folder.

The three modules are:

1. `data_pipeline` — web scraping, data cleaning, currency conversion, SQLite storage, SQL queries, and pandas analysis
2. `analytics` — Titanic EDA, classification, imbalance handling, model tuning, regression, and model saving
3. `support_assistant` — policy document embeddings, ChromaDB retrieval, LangGraph routing, FastAPI, and Docker

---

## Project Structure

```text
prem_capstone_project/

data_pipeline/
    scrape_books.py
    data_cleaning.py
    database.py
    queries.py
    books.db
    README.md
    output/

analytics/
    01_eda.py
    02_modeling.py
    titanic.csv
    best_model_pipeline.joblib
    README.md
    outputs/

support_assistant/
    docs/
    ingest.py
    graph.py
    schemas.py
    main.py
    Dockerfile
    README.md
    chroma_db/

README.md
requirements.txt
```

---

# Requirements

This project uses one common `requirements.txt` file in the root folder for all three modules.

Install the required packages using:

```bash
pip install -r requirements.txt
```

The main libraries used in the project include:

```text
requests
beautifulsoup4
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
sentence-transformers
chromadb
langgraph
fastapi
uvicorn
pydantic
```

Python standard-library packages such as `sqlite3`, `pathlib`, and `os` are not included in `requirements.txt`.

---

# Module 1 — Data Pipeline

## Purpose

The first module builds a simple product-data pipeline using the Books to Scrape website.

The workflow is:

```text
Scrape
→ Clean
→ Convert
→ Store
→ Query
```

The scraper collects:

* Title
* Price in GBP
* Star rating
* Availability
* Category

The price is converted using the project-defined rate:

```text
1 GBP = 105.50 INR
```

The cleaned data is stored in a normalized SQLite database with `books` and `categories` tables.

SQL and pandas are then used to query and compare the data.

## Run Module 1

From the project root:

```bash
python3 data_pipeline/scrape_books.py
```

```bash
python3 data_pipeline/data_cleaning.py
```

```bash
python3 data_pipeline/database.py
```

```bash
python3 data_pipeline/queries.py
```

More details are available in:

```text
data_pipeline/README.md
```

---

# Module 2 — Analytics Pipeline

## Purpose

The second module uses the Titanic dataset for exploratory data analysis and machine learning.

The workflow includes:

```text
Load Data
→ Profile Data
→ Clean Data
→ EDA
→ Train/Test Split
→ Preprocessing
→ Classification
→ Evaluation
→ Imbalance Handling
→ Hyperparameter Tuning
→ Regression
→ Save Model
```

Three classification models are trained:

* Logistic Regression
* Decision Tree
* Random Forest

The best overall classifier from the main comparison was Random Forest.

Its main test results were:

```text
Accuracy  = 0.8212
Precision = 0.8136
Recall    = 0.6957
F1        = 0.7500
AUC       = 0.8300
```

The regression task predicts passenger fare using Linear Regression.

Its results were:

```text
MAE         = 20.8094
RMSE        = 30.4731
R²          = 0.3999
Adjusted R² = 0.3679
```

The final classification pipeline is saved as:

```text
analytics/best_model_pipeline.joblib
```

## Run Module 2

Run the EDA stage:

```bash
python3 analytics/01_eda.py
```

Then run modeling:

```bash
python3 analytics/02_modeling.py
```

More details are available in:

```text
analytics/README.md
```

---

# Module 3 — Support Assistant

## Purpose

The third module builds a simple Zepto policy support assistant.

The workflow is:

```text
Policy Documents
→ Embeddings
→ ChromaDB
→ LangGraph
→ FastAPI
→ Docker
```

The module uses 8 Zepto policy documents.

Embeddings are generated locally using:

```text
all-MiniLM-L6-v2
```

and stored in ChromaDB.

LangGraph uses three main nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

Policy questions are routed to ChromaDB retrieval.

General questions return a fixed response.

The graded version uses mock mode and does not require a paid LLM API.

## Run Document Ingestion

```bash
python3 support_assistant/ingest.py
```

## Test LangGraph

```bash
python3 support_assistant/graph.py
```

## Run FastAPI

```bash
MOCK_LLM=1 uvicorn support_assistant.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run with Docker

Build the Docker image:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
```

Run the container:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

Open:

```text
http://127.0.0.1:7860/docs
```

More details are available in:

```text
support_assistant/README.md
```

---

# Final Summary

This project covers an end-to-end data and AI workflow.

Module 1 focuses on collecting and storing structured data.

Module 2 focuses on analyzing data and building machine-learning models.

Module 3 focuses on retrieval, routing, API development, and containerization.

Together, the three modules demonstrate the complete flow from raw data collection to analytics, prediction, and a simple AI-powered support service.
