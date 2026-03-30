# Technical Presentation: Sound Realty Pricing Engine Architecture

**Duration:** ~15 minutes | **Audience:** Engineers & Scientists

---

## Slide 1: Architecture Overview (1.5 min)

**System Components:**
```
REST API (FastAPI)
    ↓
RandomForest Model (18 features)
    ↓
Demographic Data Store
    ↓
Prediction Cache Layer (optional)
```

**Tech Stack:**
- **Framework**: FastAPI 1.0 (async Python web framework)
- **Model**: scikit-learn RandomForestRegressor
- **Data**: pandas (feature engineering), numpy (numerical ops)
- **Deployment**: Docker + Docker Compose (multi-service orchestration)
- **Testing**: pytest with TestClient (in-process FastAPI testing)
- **Language**: Python 3.10.14

**Why These Choices:**
- FastAPI: Async I/O, automatic API documentation (OpenAPI/Swagger), fast JSON serialization
- RandomForest: Non-linear relationships, robust to outliers, explainable feature importance
- Docker: Reproduce environment anywhere, scale independently

---

## Slide 2: Model Development & Performance (2 min)

**Initial Model (Baseline):**
- Algorithm: KNeighborsRegressor (k=10)
- Holdout R²: 0.728
- RMSE: $164K
- **Problem**: Underfitting, high bias

**Improved Model (Current):**
- Algorithm: RandomForestRegressor (400 estimators)
- Hyperparameters:
  - max_features: 0.6
  - max_depth: None (auto)
  - min_samples_leaf: 1
- Holdout R²: **0.885** (+21.6% vs baseline)
- Holdout RMSE: **$131K** (-20.0% vs baseline)
- Holdout MAPE: **12.8%**
- Holdout coverage: **55.5% within 10%** and **81.6% within 20%**

**Hyperparameter Optimization:**
- Method: RandomizedSearchCV (12 iterations, 3-fold cross-validation)
- Objective: Minimize RMSE
- Results saved to `model/hpo_results.json` for reproducibility
- CV-optimized params outperformed defaults

**Feature Set:**
- **Original 8 features** (required): bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, zipcode
- **Extended 10 features** (optional with learned defaults): waterfront, view, condition, grade, yr_built, yr_renovated, lat, long, sqft_living15, sqft_lot15
- Defaults computed as medians from training data (backward compatible)

---

## Slide 3: Training Pipeline (1.5 min)

**Data Source:**
- Primary: `kc_house_data.csv` (21,613 homes, 21 columns)
- Secondary: `zipcode_demographics.csv` (79 zipcodes, 5 demographic features)
- Test: `future_unseen_examples.csv` (100 unseen homes, no price labels)

**Training Workflow:**
```python
1. Load kc_house_data.csv
2. Join demographic features on zipcode
3. Select 18 features (original 8 + extended 10)
4. 80/20 train/holdout split (stratified optional)
5. StandardScaler normalization
6. RandomizedSearchCV(RandomForest, param_grid, cv=3)
7. Train winner on full training set
8. Evaluate on holdout set
9. Save artifacts:
   - model.pkl (model binary)
   - model_features.json (feature order)
   - test_metrics.json (train + holdout metrics)
   - hpo_results.json (hyperprameters)
   - input_feature_defaults.json (optional field medians)
```

**Metrics Computed:**
- RMSE, MAE, R², Median Absolute Error
- MAPE (Mean Absolute Percentage Error)
- Bias, WAPE (Weighted Absolute Percentage Error)
- pct_within_10 (% predictions within 10% of actual price)
- pct_within_20 (% predictions within 20% of actual price)

---

## Slide 4: API Design & Inference (1.5 min)

**Endpoint: `POST /predict/batch`**

**Request Payload:**
```json
[
  {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2500,
    "sqft_lot": 5000,
    "floors": 2,
    "sqft_above": 1800,
    "sqft_basement": 700,
    "zipcode": "98109",
    "waterfront": null,
    "view": 2,
    "condition": 4
  }
]
```

**Response:**
```json
{
  "predicted_prices": [562000.00, 485000.00, ...]
}
```

**Inference Logic:**
1. **Validate** JSON schema (required fields present, types correct)
2. **Validate** zipcode exists in demographics lookup
3. **Join** demographic data on zipcode (backend enrichment)
4. **Impute** optional fields: if null, use median from `input_feature_defaults.json`
5. **Reorder** features to match training order (critical for tree-based models)
6. **Predict** using model.pkl
7. **Return** JSON array of prices

**Design Decisions:**
- Batch endpoint (not single): Amortizes JSON parsing, allows vectorized operations
- Backend enrichment: Clients don't need demographics; we add it server-side
- Optional fields with learned defaults: Better than hardcoded 0 (preserves signal)
- No quality thresholds: Return all predictions, possible checks in the future.

---

## Slide 5: Testing Strategy (1.5 min)

**Test Suite (10 tests, all passing):**

| Test | Purpose | Coverage |
|------|---------|----------|
| `test_health_check` | Verify API is alive | Endpoint correctness |
| `test_batch_single_home` | Single prediction works | Basic functionality |
| `test_batch_multiple_homes` | Batch size matched | Array handling |
| `test_batch_original_minimal_columns` | 8-field payload scores | Backward compatibility |
| `test_batch_optional_fields_accepted` | Extra fields don't break | Forward compatibility |
| `test_batch_empty_list_returns_422` | Empty array rejected | Input validation |
| `test_batch_missing_required_field` | Missing field rejected | Schema enforcement |
| `test_batch_unknown_zipcode` | Bad zipcode rejected | Data integrity |
| `test_batch_scores_unseen_dataset_fast` | 100 rows in <5s | Performance budget |
| `test_reports_held_metrics_valid` | Metrics structure correct | Model artifact validation |

**Testing Approach:**
- **TestClient** with module scope (single lifespan for all tests)
- **Fixture-based**: Reusable client, clean state per test
- **End-to-end**: Tests hit real API handlers, real model
- **Edge cases**: Empty lists, missing fields, invalid zipcodes
- **Performance**: Currently only checks if the metrics are properly calculated, can be used in the future to verify retraining.

---

## Slide 6: Deployment Architecture (1.5 min)

**Docker Compose Services:**

```yaml
services:
  train:
    build: .
    command: python create_model.py
    volumes:
      - ./data:/app/data         # input: training data
      - ./model:/app/model       # output: artifacts
    profiles: [train]
    
  test:
    build: .
    command: pytest tests/test_api.py -v
    volumes:
      - ./model:/app/model       # consume artifacts
    profiles: [test]
    
  api:
    build: .
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data         # demographics CSV
      - ./model:/app/model       # model artifacts
```

**Workflow:**
```
$ docker-compose --profile train up train  # Train
$ docker-compose --profile test up test    # Test
$ docker-compose up api                    # Serve
```

**Lifespan Management:**
- Async context manager in FastAPI
- Load model.pkl, model_features.json, demographics CSV once at startup
- Kept in memory for all requests (zero latency for repeated predictions)
- Clean shutdown (proper resource cleanup)