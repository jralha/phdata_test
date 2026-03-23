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

## API Endpoints

- `GET /health` - health check
- `POST /predict/batch` - batch home price predictions

### `/predict/batch` payload shape

The endpoint expects a JSON array of property objects. Each object should use
the columns from `data/future_unseen_examples.csv`.

```json
[
	{
		"bedrooms": 4,
		"bathrooms": 1.0,
		"sqft_living": 1680,
		"sqft_lot": 5043,
		"floors": 1.5,
		"waterfront": 0,
		"view": 0,
		"condition": 4,
		"grade": 6,
		"sqft_above": 1680,
		"sqft_basement": 0,
		"yr_built": 1911,
		"yr_renovated": 0,
		"zipcode": "98118",
		"lat": 47.5354,
		"long": -122.273,
		"sqft_living15": 1560,
		"sqft_lot15": 5765
	}
]
```