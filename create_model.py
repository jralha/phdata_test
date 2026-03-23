import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

DATA_DIR = "data"
SALES_PATH = f"{DATA_DIR}/kc_house_data.csv"
DEMOGRAPHICS_PATH = f"{DATA_DIR}/zipcode_demographics.csv"
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'waterfront', 'view', 'condition',
    'grade', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15',
    'sqft_lot15', 'zipcode'
]
OUTPUT_DIR = "model"
OPTIONAL_INPUT_COLUMNS = [
    "waterfront",
    "view",
    "condition",
    "grade",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]


def train_model_with_hpo(x_train: pandas.DataFrame, y_train: pandas.Series):
    """Train a RandomForestRegressor using a small randomized CV search."""
    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [200, 400, 700],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.6, 1.0],
    }

    search = model_selection.RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=12,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, -search.best_score_


def calculate_metrics(y_true: pandas.Series, y_pred, split: str) -> List[dict]:
    """Calculate regression metrics for a given data split."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    wape = (y_true - y_pred).abs().sum() / y_true.abs().sum()
    pct_within_10 = ((y_true - y_pred).abs() / y_true <= 0.10).mean()
    pct_within_20 = ((y_true - y_pred).abs() / y_true <= 0.20).mean()

    return [
        {"split": split, "metric": "rmse", "value": float(rmse)},
        {"split": split, "metric": "mae", "value": float(mae)},
        {"split": split, "metric": "r2", "value": float(r2)},
        {"split": split, "metric": "medae", "value": float(medae)},
        {"split": split, "metric": "mape", "value": float(mape)},
        {"split": split, "metric": "bias", "value": float(bias)},
        {"split": split, "metric": "wape", "value": float(wape)},
        {
            "split": split,
            "metric": "pct_within_10",
            "value": float(pct_within_10),
        },
        {
            "split": split,
            "metric": "pct_within_20",
            "value": float(pct_within_20),
        },
    ]


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with demographics data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containing two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        Series contains the target variable (home sale price).
    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path,
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def compute_input_feature_defaults(data: pandas.DataFrame) -> dict:
    """Return median defaults for optional input fields."""
    defaults = {}
    for col in OPTIONAL_INPUT_COLUMNS:
        defaults[col] = float(data[col].median())
    return defaults


def main():
    """Load data, train model, and export artifacts."""
    sales_data = pandas.read_csv(
        SALES_PATH,
        usecols=SALES_COLUMN_SELECTION,
        dtype={"zipcode": str},
    )
    input_feature_defaults = compute_input_feature_defaults(sales_data)
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42)

    model, best_params, cv_rmse = train_model_with_hpo(x_train, y_train)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features.json", 'w'))
    with open(output_dir / "input_feature_defaults.json", "w") as f:
        json.dump(input_feature_defaults, f, indent=2)

    print(f"Model artifacts written to {output_dir.resolve()}")

    train_metrics = calculate_metrics(y_train, model.predict(x_train), "train")
    holdout_metrics = calculate_metrics(y_test, model.predict(x_test), "holdout")
    metrics = train_metrics + holdout_metrics

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    hpo_report = {
        "model": "RandomForestRegressor",
        "search": "RandomizedSearchCV",
        "best_params": best_params,
        "best_cv_rmse": float(cv_rmse),
    }
    with open(output_dir / "hpo_results.json", "w") as f:
        json.dump(hpo_report, f, indent=2)

    print("Saved metrics to model/test_metrics.json")
    print("Saved HPO report to model/hpo_results.json")


if __name__ == "__main__":
    main()
