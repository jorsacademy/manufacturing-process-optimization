from pathlib import Path

import pandas as pd

from leadtime_ml.features import TARGET, validate_features
from leadtime_ml.modeling import build_pipeline, load_pipeline, save_pipeline


def test_pipeline_round_trip(tmp_path: Path):
    frame = pd.DataFrame(
        [
            [100, 0.60, 3, 30, 4, 95, 3, "A", "day", "M1", 24.0],
            [220, 0.80, 8, 60, 2, 82, 2, "B", "night", "M3", 49.0],
            [160, 0.72, 5, 40, 8, 97, 5, "C", "evening", "M2", 32.0],
            [320, 0.90, 12, 75, 3, 76, 1, "D", "night", "M4", 69.0],
            [90, 0.55, 2, 25, 10, 99, 4, "A", "day", "M1", 18.0],
            [260, 0.84, 7, 55, 5, 88, 2, "B", "evening", "M2", 45.0],
        ],
        columns=[
            "order_quantity",
            "machine_utilization",
            "queue_length",
            "setup_minutes",
            "operator_experience_years",
            "material_availability_pct",
            "priority_score",
            "product_family",
            "shift",
            "machine_group",
            TARGET,
        ],
    )

    X = validate_features(frame)
    y = frame[TARGET]

    pipeline = build_pipeline()
    pipeline.set_params(model__n_estimators=10)
    pipeline.fit(X, y)

    path = tmp_path / "model.joblib"
    save_pipeline(pipeline, path)
    loaded = load_pipeline(path)

    predictions = loaded.predict(X)
    assert len(predictions) == len(frame)
