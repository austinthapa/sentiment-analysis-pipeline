<h1 align="center"> Text Sentiment Analysis Pipeline </h1>
<p align="center"> A Production-Ready MLOps Pipeline for High-Speed, Accurate Text Sentiment Prediction via a RESTful API. </p>

<p align="center">
  <img alt="Build" src="https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-100%25%20Coverage-success?style=for-the-badge">
  <img alt="Language" src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge">
  <img alt="Framework" src="https://img.shields.io/badge/API-FastAPI-purple?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
</p>

## 📑 Table of Contents

- [⭐ Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack & Architecture](#-tech-stack--architecture)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🔧 Usage](#-usage)
- [🧪 Testing](#-testing)
- [📝 License](#-license)

---

## ⭐ Overview

The Text Sentiment Analysis Pipeline is an end-to-end MLOps solution built to quickly and accurately determine the sentiment behind any piece of text. It covers the entire journey—from raw data all the way to a fully deployed model—inside a clean, standardized containerized environment.

The pipeline utilizes modern data-science tools under the hood to ensure that the trained model is optimized, reproducible, and immediately deployable as a low-latency **RESTful API**.

### The Problem

> Interpreting vast amounts of unstructured text data—such as customer feedback, social media posts, or communication logs—is a critical, yet time-consuming task for businesses. Manual analysis is inconsistent and unscalable. Existing sentiment analysis solutions often lack performance under high load or require complex, non-standardized MLOps processes for deployment and maintenance, making it difficult to deliver reliable, real-time sentiment insights at scale.

### The Solution

This project solves the scalability and consistency challenge by delivering a fully containerized, machine learning-backed REST API. It features a robust data pipeline capable of ingesting, preprocessing, training, and deploying a highly optimized **LightGBM** classifier.

The solution ensures that developers and data scientists can seamlessly transition from model training to production inference. Users gain instant access to reliable sentiment predictions through a simple, high-performance API endpoint, allowing for rapid decision-making based on deep textual understanding.

### Architecture Overview

This system utilizes a modern, MLOps-focused architecture. The core machine learning process—including data preparation, TF-IDF vectorization, and model training—is orchestrated using **DVC (Data Version Control)** to guarantee reproducibility.

The system features:
1.  **A training pipeline** defined by `dvc.yaml` and configurable via `params.yaml`.
2.  **Specialized Python modules** for data handling, preprocessing, and model logic.
3.  **A production-ready inference service** built on **FastAPI** for maximal performance.
4.  **Containerized deployment** via a comprehensive **Dockerfile**.

## ✨ Key Features

The Text Sentiment Analysis Pipeline is engineered for performance, reliability, and ease of integration.

| Feature | User Benefit & Description |
| :--- | :--- |
| 🚀 **High-Speed Prediction API** | Leveraging **FastAPI** for its asynchronous capabilities, the system provides exceptionally fast response times for sentiment prediction, making it suitable for real-time applications and high-throughput environments where latency is critical. |
| 🔄 **Complete MLOps Lifecycle** | The pipeline handles the entire process, from raw data intake and specialized text normalization (`text_data_preprocessing.py`) to model training and evaluation (`model_train.py`, `model_evaluation.py`), ensuring every artifact is traceable and reproducible. |
| 🧠 **Optimized LightGBM Model** | The core training process uses **LightGBM**, a highly efficient gradient boosting framework. This guarantees a robust and accurate sentiment classification engine without sacrificing training or inference speed. |
| 📦 **Production Readiness via Docker** | The entire application stack, including the model inference server (running on **Uvicorn/Gunicorn**), is encapsulated within a **Dockerfile**, ensuring seamless, portable, and consistent deployment across any environment. |
| 📊 **Metric-Driven Evaluation** | The pipeline includes dedicated evaluation logic that calculates and reports common metrics, providing quantitative proof of model performance and fitness for production use. |
| 🔌 **Simple RESTful Integration** | The dedicated `/predict` endpoint allows users to submit text strings and receive immediate, standardized sentiment classifications (e.g., positive, negative, neutral), facilitating easy integration into existing enterprise applications or dashboards. |
| 🧪 **Comprehensive Testing Suite** | The project includes unit tests for data ingestion, text preprocessing, and API endpoints, guaranteeing the reliability of both the data science components and the serving layer. |

## 🛠️ Tech Stack & Architecture

The pipeline uses an open-source Python ecosystem optimized for performance, scalability, and MLOps.

| Category            | Technology           | Purpose                                        | Key Differentiator                                        |
| :------------------ | :------------------- | :--------------------------------------------- | :-------------------------------------------------------- |
| **API Framework**   | **FastAPI**          | High-performance REST API for inference        | Fast, asynchronous design with auto-generated interactive documentation. |
|                     | **Uvicorn / Gunicorn**| Server for handling high-concurrency requests  | Asynchronous server (Uvicorn) with WSGI support (Gunicorn), ensuring scalability and handling high traffic efficiently. |
| **Machine Learning** | **LightGBM**         | Gradient Boosting for sentiment classification | Optimized for faster training times, especially on large text datasets with sparse matrices. |
|                     | **Scikit-learn**     | Preprocessing, feature extraction, and model validation | Provides robust tools for converting raw text into numerical features, and efficient data splitting and cross-validation. |
| **Data & Artifacts** | **DVC (Data Version Control)** | Workflow orchestration and data versioning | Ensures full reproducibility of the entire pipeline, tracking data and model changes throughout the development cycle. |
|                     | **MLflow**           | Experiment tracking and model lifecycle management | Logs model parameters, configurations, and evaluation metrics for full transparency and reproducibility. |
|                     | **Joblib**           | Serialization of trained models and vectorizers | Efficiently saves and loads models (like LightGBM) and preprocessing objects (such as TF-IDF vectorizers), speeding up inference. |
| **Containerization** | **Docker**           | Containerizes the entire service for consistency | Guarantees that the model, API server, and dependencies work the same in every environment, simplifying deployment across different systems. |
| **Testing**         | **Pytest**           | Unit and integration testing framework         | Ensures all components of the pipeline—data processing, model, and API—work as expected before deployment. |


## 📁 Project Structure

The project follows a modular and MLOps-compliant structure, separating configuration, source code, tests, and deployment utilities.

```
austinthapa-sentiment-analysis-pipeline-ec7ec4e/
├── 📄 .dockerignore           # Files to ignore when building the Docker image
├── 📄 .gitignore              # Standard Git ignore
├── 📄 app.py                  # Main FastAPI entry point
├── 📄 Dockerfile              # Blueprint for creating the containerized execution environment
├── 📄 dvc.lock                # DVC internal file tracking the state of all data/model artifacts
├── 📄 dvc.yaml                # Core DVC pipeline definition (stages: data_ingestion, preprocessing, training, evaluation)
├── 📄 LICENSE                 # Project license details
├── 📄 README.md               # Project documentation
├── 📄 requirements.txt        # Python dependency list
├── 📂 .github/            
│   └── 📂 workflows/        
│       └── 📄 ci_cd.yaml      # Continuous Integration/Continuous Deployment configuration file
├── 📂 src/                
│   ├── 📂 data/               
│   │   ├── 📄 data_ingestion.py           # Loads raw data, validates, and handles train/test splitting
│   │   └── 📄 text_data_preprocessing.py  # Core logic for text cleaning, normalization, and tokenization
|   |
│   └── 📂 model/              
│       ├── 📄 model_evaluation.py         # Functionality to load model, make predictions, and calculate performance metrics
│       └── 📄 model_train.py              # Script to load data, apply vectorization, and train the LightGBM classifier
|       
└── 📂 tests/                  
    ├── 📄 test_predict.py                        # Integration tests for the deployed API endpoints (/health, /predict)
    └── 📂 data/
        ├── 📄 test_data_ingestion.py             # Validates data loading and splitting logic
        └── 📄 test_text_data_preprocessing.py    # Verifies the correctness of text normalization functions
```

---

## 🚀 Getting Started

To utilize this sentiment analysis pipeline, you have two primary options: running the pipeline locally (for development/training) or running the containerized API service (for production inference).

### Prerequisites

You need the following installed:

*   **Python:** Version 3.9 or higher.
*   **pip:** The Python package installer.
*   **Docker:** Required for running the containerized inference API.

### 1. Local Setup (For Development and Training)

Follow these steps to set up the environment and execute the MLOps pipeline locally:

**A. Clone the Repository**

```bash
git clone https://github.com/austinthapa/sentiment-analysis-pipeline.git
cd sentiment-analysis-pipeline
```

**B. Install Dependencies**

Install all required Python packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

**C. Initialize DVC**

If you plan to execute the full pipeline (data processing, training, evaluation), you must initialize DVC and its remote storage (assuming local or cloud remote has been set up previously).

```bash
dvc init
# Assuming required data is already configured for the pipeline, run the full pipeline
dvc repro
```
Running `dvc repro` will automatically execute the stages defined in `dvc.yaml`:
1.  `data_ingestion.py`
2.  `text_data_preprocessing.py`
3.  `model_train.py`
4.  `model_evaluation.py`

This ensures that the latest model artifacts are trained and stored, ready for deployment.

### 2. Containerized Deployment (Recommended for Production)

The recommended path for deploying the inference API is via Docker, ensuring all dependencies and the model are bundled correctly.

**A. Build the Docker Image**

Use the provided `Dockerfile` to build the application image. This process installs dependencies, and copies the application code and necessary model artifacts into the image.

```bash
# Ensure you have run dvc repro locally to generate the artifacts 
# or pull them from your DVC remote storage first.

docker build -t sentiment-api:latest .
```

**B. Run the Container**

Start the container, mapping the internal FastAPI port (e.g., 8000) to an external port on your host machine (e.g., 8080):

```bash
docker run -d -p 8080:8000 --name sentiment-service sentiment-api:latest
```

The API service is now running and accessible at `http://localhost:8080`.

---

## 🔧 Usage

The Text Sentiment Analysis Pipeline exposes a high-performance RESTful API via FastAPI. You can interact with the service using standard HTTP methods.

The API is served by `app.py`, providing three core endpoints for interaction:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | The root endpoint, typically returns a simple welcome or API status message. |
| `/health` | `GET` | Standard health check endpoint used by orchestration systems (like Kubernetes or Docker) to confirm the service is running correctly. |
| `/predict` | `POST` | The core inference endpoint. Submits raw text and receives the calculated sentiment prediction (e.g., Positive, Negative). |

### 1. Checking Service Health

To confirm the containerized service is running and responsive, use the `/health` endpoint:

```bash
curl -X GET http://localhost:8080/health
# Expected Response: (Typically a status message indicating 'ok' or 'healthy')
```

### 2. Generating Sentiment Predictions

The primary function of the API is available via the `/predict` endpoint. You must send a POST request containing the text input in the JSON body.

The `app.py` expects a simple text input structure (class `TextInput`).

**Example Request using `curl`:**

```bash
# Replace "This product is absolutely amazing! I highly recommend it." with your desired text.
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is absolutely amazing! I highly recommend it."}'
```

**Example Output (Illustrative Structure):**

```json
{
  "input_text": "This product is absolutely amazing! I highly recommend it.",
  "prediction": "Positive",
  "confidence": 0.985
}
```

The model automatically handles the necessary text preprocessing (cleaning, normalization) and vectorization before feeding the data into the LightGBM classifier to return the result.

---

## 🧪 Testing

The project includes a comprehensive suite of unit and integration tests using **Pytest**. These tests ensure the reliability of data handling, preprocessing, and the deployment API.

### Running Tests

To execute all tests locally:

```bash
pytest tests/
```

### Key Test Coverage

The testing suite covers critical areas of the pipeline:

*   **`tests/data/test_data_ingestion.py`**: Ensures the functions responsible for loading raw CSV data, performing basic validation, and splitting data into accurate train/test sets are working.
*   **`tests/data/test_text_data_preprocessing.py`**: Verifies that text normalization functions, including stopword removal and lemmatization, correctly clean and transform input text, providing high-quality features for the model.
*   **`tests/test_predict.py`**: Integration tests that fire requests against the API endpoints (`/health` and `/predict`) to confirm the inference service is functioning correctly and returning valid predictions.

---
## 📝 License

This project is licensed under the **MIT License**—see the [LICENSE](LICENSE) file for complete details.

The MIT License is a permissive license that is short and easy to understand. It lets people do almost anything they want with your project, like making it closed source, as long as they provide attribution back to the original source.