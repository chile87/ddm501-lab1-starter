# Lab 1: First ML Product - Movie Rating Prediction API

## Overview

Build your first ML product - a Movie Rating Prediction API using collaborative filtering, REST API, and Docker containerization.

## Project Structure

```
ddm501-lab1-starter/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application (TODO)
│   ├── model.py          # ML model loading & prediction (TODO)
│   ├── schemas.py        # Pydantic models (TODO)
│   └── config.py         # Configuration
├── models/               # Saved ML models
├── tests/
│   ├── __init__.py
│   └── test_api.py       # Unit tests (TODO)
├── scripts/
│   └── train_model.py    # Model training script
├── Dockerfile            # (TODO)
├── docker-compose.yml    # (TODO)
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

## Quick Start

### 1. Clone and Setup

```bash
unzip ddm501-lab1-starter.zip
cd ddm501-lab1-starter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python scripts/train_model.py
```

This will download MovieLens 100K dataset and train an SVD model.

### 3. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "196", "movie_id": "242"}'
```

### 5. Run with Docker

```bash
docker-compose build
docker-compose up -d
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Get rating prediction |
| GET | `/docs` | Swagger documentation |

## TODO Tasks

Complete the following files:

- [ ] `app/model.py` - Implement `load_model()`, `predict()`, `predict_batch()`
- [ ] `app/schemas.py` - Define Pydantic request/response models
- [ ] `app/main.py` - Implement `/predict` endpoint with error handling
- [ ] `Dockerfile` - Complete with health check
- [ ] `docker-compose.yml` - Configure services
- [ ] `tests/test_api.py` - Add edge case tests

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html
```

## Grading Rubric

| Criteria | Weight |
|----------|--------|
| Working ML Model | 25% |
| REST API | 25% |
| Docker Setup | 20% |
| Test Cases | 20% |
| Documentation | 10% |

## Submission

1. Complete all TODO tasks
2. Ensure all tests pass
3. Push to your GitHub repository
4. Submit the repository link
