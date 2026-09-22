from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from leadtime_ml.features import validate_features
from leadtime_ml.modeling import load_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic batch inference.")
    parser.add_argument("--input", default="data/incoming/orders_to_score.csv")
    parser.add_argument("--model", default="models/lead_time_pipeline.joblib")
    parser.add_argument("--output", default="data/predictions/latest_predictions.csv")
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    features = validate_features(source)
    pipeline = load_pipeline(args.model)

    predictions = pipeline.predict(features)

    output = source.copy()
    output["predicted_lead_time_hours"] = predictions.round(2)
    output["scored_at_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Scored {len(output)} orders -> {output_path}")


if __name__ == "__main__":
    main()
