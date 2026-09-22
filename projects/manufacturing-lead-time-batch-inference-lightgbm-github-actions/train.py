from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from leadtime_ml.features import TARGET, validate_features
from leadtime_ml.modeling import build_pipeline, save_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the manufacturing lead-time model.")
    parser.add_argument("--data", default="data/training.csv")
    parser.add_argument("--model-out", default="models/lead_time_pipeline.joblib")
    parser.add_argument("--metrics-out", default="models/metrics.json")
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    if TARGET not in frame.columns:
        raise ValueError(f"Training data must contain target column: {TARGET}")

    X = validate_features(frame)
    y = frame[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline(random_state=42)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = {
        "mae_hours": round(float(mean_absolute_error(y_test, predictions)), 4),
        "mape": round(float(mean_absolute_percentage_error(y_test, predictions)), 4),
        "test_rows": len(y_test),
        "mean_actual_hours": round(float(np.mean(y_test)), 4),
        "mean_prediction_hours": round(float(np.mean(predictions)), 4),
    }

    save_pipeline(pipeline, args.model_out)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
