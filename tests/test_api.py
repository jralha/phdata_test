"""Tests for the Sound Realty prediction API."""
import json
import math
from pathlib import Path
from time import perf_counter

import pandas
import pytest
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture(scope="module")
def client():
    """Start app (runs lifespan, loads model + demographics) once per module."""
    with TestClient(app) as c:
        yield c


# Test payloads
VALID_HOME = {
    "bedrooms": 4,
    "bathrooms": 1.0,
    "sqft_living": 1680,
    "sqft_lot": 5043,
    "floors": 1.5,
    "sqft_above": 1680,
    "sqft_basement": 0,
    "zipcode": "98118",
}

VALID_HOME_2 = {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2220,
    "sqft_lot": 6380,
    "floors": 1.5,
    "sqft_above": 1660,
    "sqft_basement": 560,
    "zipcode": "98115",
}

UNSEEN_DATA_PATH = Path("data/future_unseen_examples.csv")
METRICS_PATH = Path("model/test_metrics.json")

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_batch_single_home_returns_one_price(client):
    response = client.post("/predict/batch", json=[VALID_HOME])
    assert response.status_code == 200
    body = response.json()
    assert "predicted_prices" in body
    assert len(body["predicted_prices"]) == 1
    assert isinstance(body["predicted_prices"][0], float)
    assert body["predicted_prices"][0] > 0


def test_batch_multiple_homes_returns_matching_count(client):
    response = client.post("/predict/batch", json=[VALID_HOME, VALID_HOME_2])
    assert response.status_code == 200
    body = response.json()
    assert len(body["predicted_prices"]) == 2


def test_batch_optional_fields_accepted(client):
    """Extra fields that the model ignores should not cause errors."""
    home_with_extras = {**VALID_HOME, "waterfront": 1, "view": 3, "grade": 8,
                        "yr_built": 1990, "lat": 47.5, "long": -122.3}
    response = client.post("/predict/batch", json=[home_with_extras])
    assert response.status_code == 200

def test_batch_empty_list_returns_422(client):
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 422

def test_batch_missing_required_field_returns_422(client):
    incomplete = {k: v for k, v in VALID_HOME.items() if k != "sqft_living"}
    response = client.post("/predict/batch", json=[incomplete])
    assert response.status_code == 422

def test_batch_unknown_zipcode_returns_422(client):
    bad_zip = {**VALID_HOME, "zipcode": "00000"}
    response = client.post("/predict/batch", json=[bad_zip])
    assert response.status_code == 422
    assert "00000" in response.json()["detail"]


def test_batch_scores_full_unseen_dataset_within_time_budget(client):
    """Smoke-test batch inference over the provided unseen dataset."""
    unseen = pandas.read_csv(UNSEEN_DATA_PATH, dtype={"zipcode": str})

    start = perf_counter()
    response = client.post("/predict/batch", json=unseen.to_dict(orient="records"))
    elapsed = perf_counter() - start

    assert response.status_code == 200
    body = response.json()
    assert len(body["predicted_prices"]) == len(unseen)
    assert all(price > 0 for price in body["predicted_prices"])
    assert elapsed < 5.0


def test_reports_saved_holdout_metrics_if_valid():
    """Report model metrics and validate train + holdout values are sane."""
    assert METRICS_PATH.exists(), (
        "Expected model/test_metrics.json. Run training first: "
        "docker compose --profile train run --rm train"
    )

    metrics_records = json.loads(METRICS_PATH.read_text())
    assert isinstance(metrics_records, list)
    assert metrics_records, "Expected at least one metric record"

    assert all("split" in record for record in metrics_records), (
        "Each metric record must include a split field"
    )

    metrics_by_split = {}
    for record in metrics_records:
        split = record["split"]
        metric = record["metric"]
        value = float(record["value"])
        metrics_by_split.setdefault(split, {})[metric] = value

    required_metrics = {
        "rmse",
        "mae",
        "r2",
        "medae",
        "mape",
        "bias",
        "wape",
        "pct_within_10",
        "pct_within_20",
    }
    for split in ("train", "holdout"):
        assert split in metrics_by_split, f"Missing split: {split}"
        missing = required_metrics - set(metrics_by_split[split])
        assert not missing, f"Missing required metrics for {split}: {sorted(missing)}"

    # Reports metrics in test output when running with -s (useful for CI logs)
    print("Saved metrics:", json.dumps(metrics_by_split, sort_keys=True))

    for split, metrics in metrics_by_split.items():
        for name, value in metrics.items():
            assert math.isfinite(value), f"Metric {split}.{name} must be finite"

        # Basic validity checks, not quality thresholds.
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["medae"] >= 0
        assert metrics["mape"] >= 0
        assert metrics["wape"] >= 0
        assert 0 <= metrics["pct_within_10"] <= 1
        assert 0 <= metrics["pct_within_20"] <= 1
