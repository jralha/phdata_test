import json
import pathlib
import pickle
from typing import List
from typing import Tuple
from xml.parsers.expat import model

import pandas
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing

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
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"


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


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42)

    model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor()).fit(
                                       x_train, y_train)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features.json", 'w'))

    print(f"Model artifacts written to {output_dir.resolve()}")

    # Test performance on _x_test
    y_pred = model.predict(x_test)

    # Add performance metrics

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    bias = (y_pred - y_test).mean()
    wape = (y_test - y_pred).abs().sum() / y_test.abs().sum()
    pct_within_10 = ((y_test - y_pred).abs() / y_test <= 0.10).mean()
    pct_within_20 = ((y_test - y_pred).abs() / y_test <= 0.20).mean()

    # Create a DataFrame to hold the metrics
    metrics_df = pandas.DataFrame({
        "metric": [
            "rmse", "mae", "r2", "medae", "mape", "bias", "wape",
            "pct_within_10", "pct_within_20"
        ],
        "value": [
            rmse, mae, r2, medae, mape, bias, wape, pct_within_10,
            pct_within_20
        ]
    })

    # Save the metrics to a json file
    metrics_df.to_json(output_dir / "test_metrics.json", orient="records")


if __name__ == "__main__":
    main()
