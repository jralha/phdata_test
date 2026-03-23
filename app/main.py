import json
import pickle
from contextlib import asynccontextmanager
from typing import List

import pandas
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "model/model.pkl"
FEATURES_PATH = "model/model_features.json"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts and demographics table once at startup."""
    with open(MODEL_PATH, "rb") as f:
        state["model"] = pickle.load(f)

    with open(FEATURES_PATH, "r") as f:
        state["features"] = json.load(f)

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
    waterfront: float = 0
    view: float = 0
    condition: float = 0
    grade: float = 0
    yr_built: float = 0
    yr_renovated: float = 0
    lat: float = 0
    long: float = 0
    sqft_living15: float = 0
    sqft_lot15: float = 0


class BatchPredictionResponse(BaseModel):
    predicted_prices: List[float]

def _build_feature_frame(homes: List[HomeFeatures]) -> pandas.DataFrame:
    """Merge home data with demographics and return df for prediction"""
    raw = pandas.DataFrame([h.model_dump() for h in homes])
    raw["zipcode"] = raw["zipcode"].astype(str)

    merged = raw.merge(state["demographics"], how="left", on="zipcode")

    missing_zip = merged["zipcode"][merged[state["features"][0]].isna()].unique().tolist()
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
    """Return predicted sale prices for a list of properties."""
    if not homes:
        raise HTTPException(status_code=422, detail="Request body must contain at least one property.")
    features = _build_feature_frame(homes)
    prices = state["model"].predict(features).tolist()
    return BatchPredictionResponse(predicted_prices=prices)
