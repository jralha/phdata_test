import json
import pickle
from contextlib import asynccontextmanager
from typing import List
from typing import Optional

import pandas
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "model/model.pkl"
FEATURES_PATH = "model/model_features.json"
INPUT_FEATURE_DEFAULTS_PATH = "model/input_feature_defaults.json"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
PAYLOAD_SHAPE_DOC = (
    "Request body must be a JSON array of home records. Each record should "
    "match the columns from data/future_unseen_examples.csv."
)
PAYLOAD_EXAMPLE = [
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
        "sqft_lot15": 5765,
    }
]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts and demographics table once at startup."""
    with open(MODEL_PATH, "rb") as f:
        state["model"] = pickle.load(f)

    with open(FEATURES_PATH, "r") as f:
        state["features"] = json.load(f)

    try:
        with open(INPUT_FEATURE_DEFAULTS_PATH, "r") as f:
            state["input_feature_defaults"] = json.load(f)
    except FileNotFoundError:
        state["input_feature_defaults"] = {}

    state["demographics"] = pandas.read_csv(
        DEMOGRAPHICS_PATH, dtype={"zipcode": str}
    )

    yield

    state.clear()


app = FastAPI(
    title="Sound Realty – Home Price Prediction API",
    description="Predicts Seattle home sale prices using a KNN regression model.",
    version="1.0.0",
    lifespan=lifespan,
)

class HomeFeatures(BaseModel):
    """Input features for a single home.

    All columns present in future_unseen_examples.csv are accepted.
    Only the subset used during training will be forwarded to the model.
    """

    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: str
    # The following columns are accepted for API compatibility but are not
    # used by the current model.
    waterfront: Optional[float] = None
    view: Optional[float] = None
    condition: Optional[float] = None
    grade: Optional[float] = None
    yr_built: Optional[float] = None
    yr_renovated: Optional[float] = None
    lat: Optional[float] = None
    long: Optional[float] = None
    sqft_living15: Optional[float] = None
    sqft_lot15: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    predicted_prices: List[float]

def _build_feature_frame(homes: List[HomeFeatures]) -> pandas.DataFrame:
    """Merge home data with demographics and return df for prediction"""
    raw = pandas.DataFrame([h.model_dump() for h in homes])
    raw["zipcode"] = raw["zipcode"].astype(str)

    for col, default in state["input_feature_defaults"].items():
        if col in raw.columns:
            raw[col] = raw[col].fillna(default)

    merged = raw.merge(state["demographics"], how="left", on="zipcode")

    missing_mask = merged[state["features"]].isnull().any(axis=1)
    missing_zip = merged.loc[missing_mask, "zipcode"].unique().tolist()
    if missing_zip:
        raise HTTPException(
            status_code=422,
            detail=f"No demographic data found for zipcode(s): {missing_zip}",
        )

    return merged[state["features"]]

@app.get("/health", tags=["Ops"])
def health_check():
    return {"status": "ok"}


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(homes: List[HomeFeatures]):
    """Return predicted sale prices for a list of properties.

    Payload shape:
    - JSON array of objects (not a single object)
    - Each object follows the fields from `data/future_unseen_examples.csv`

    Example payload:
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
    """
    if not homes:
        raise HTTPException(status_code=422, detail="Request body must contain at least one property.")
    features = _build_feature_frame(homes)
    prices = state["model"].predict(features).tolist()
    return BatchPredictionResponse(predicted_prices=prices)
