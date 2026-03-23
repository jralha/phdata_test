"""Submit examples from data/future_unseen_examples.csv to the batch endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pandas

DEFAULT_API_URL = "http://localhost:8000/predict/batch"
DEFAULT_INPUT_CSV = Path("data/future_unseen_examples.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send unseen example rows to the batch prediction endpoint and "
            "print a compact summary."
        )
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Batch endpoint URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"CSV path to submit (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of rows to send. 0 means all rows.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    unseen = pandas.read_csv(args.input_csv, dtype={"zipcode": str})
    if args.limit > 0:
        unseen = unseen.head(args.limit)

    payload = unseen.to_dict(orient="records")
    if not payload:
        raise SystemExit("Input CSV produced an empty payload.")

    with httpx.Client(timeout=args.timeout_seconds) as client:
        response = client.post(args.api_url, json=payload)

    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}")
        print(response.text)
        raise SystemExit(1)

    body = response.json()
    predictions = body.get("predicted_prices", [])

    if len(predictions) != len(payload):
        print(
            "Warning: prediction count does not match payload size "
            f"({len(predictions)} vs {len(payload)})"
        )

    preview_count = min(5, len(predictions))
    summary = {
        "submitted_rows": len(payload),
        "received_predictions": len(predictions),
        "preview_predictions": predictions[:preview_count],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
