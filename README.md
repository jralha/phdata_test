# phdata Home Price Prediction API

This repository contains a FastAPI service for batch home price prediction, a training script for the underlying scikit-learn model, and tests for the API.

## Project Layout

- `app/` - FastAPI application
- `create_model.py` - model training script
- `data/` - training and inference input data
- `model/` - generated model artifacts
- `tests/` - API test suite
- `legacy/` - archived candidate materials and original project files

## Docker Workflow

### 1. Train the model

This writes `model/model.pkl` and `model/model_features.json` to the host.

```sh
docker compose --profile train run --rm train
```

### 2. Run the tests in the container

This executes the pytest suite inside Docker using the trained model artifacts from `./model`.

```sh
docker compose --profile test run --rm test
```

### 3. Start the API

```sh
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## Local Workflow

If you prefer to run locally instead of Docker:

```sh
python -m pip install -r requirements.txt
python create_model.py
python -m pytest tests/test_api.py -v
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /health` - health check
- `POST /predict/batch` - batch home price predictions