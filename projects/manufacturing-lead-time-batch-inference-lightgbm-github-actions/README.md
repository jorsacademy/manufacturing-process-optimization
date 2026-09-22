# Manufacturing Lead-Time Batch Inference with LightGBM and GitHub Actions

A small production-style machine-learning project that predicts manufacturing order lead time in batch mode.

Unlike a notebook-only demo, the repository separates data generation, model training, validation, inference, tests, and automation. The complete preprocessing + LightGBM estimator is persisted as one scikit-learn `Pipeline`, reducing train/serve skew.

## Architecture

```text
data/training.csv
        |
        v
    train.py
        |
        v
models/lead_time_pipeline.joblib
        |
        +-----------------------+
                                |
data/incoming/orders_to_score.csv
                                |
                                v
                            score.py
                                |
                                v
data/predictions/latest_predictions.csv
```

GitHub Actions runs CI on pushes/PRs and can run the batch scoring workflow every Monday.

## Project structure

```text
.
├── .github/
│   └── workflows/
│       ├── batch-inference.yml
│       └── ci.yml
├── data/
│   ├── incoming/
│   └── predictions/
├── models/
├── src/
│   └── leadtime_ml/
│       ├── __init__.py
│       ├── features.py
│       └── modeling.py
├── tests/
│   ├── test_features.py
│   └── test_pipeline.py
├── generate_demo_data.py
├── score.py
├── train.py
└── pyproject.toml
```

## Local run

Python 3.13 is used in CI.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"

python generate_demo_data.py
python train.py
python score.py

pytest -q
ruff check .
```

The prediction output is written to:

```text
data/predictions/latest_predictions.csv
```

Training metrics are written to:

```text
models/metrics.json
```

## Why this design?

- One serialized pipeline contains both preprocessing and the LightGBM model.
- Batch inference validates the input schema before predicting.
- Unknown categorical values are handled safely.
- Predictions are a machine-readable CSV instead of unstructured log text.
- CI is read-only.
- Scheduled inference publishes outputs as a GitHub Actions artifact instead of committing generated files back into `main`.
- `workflow_dispatch` allows manual runs.
- `concurrency` prevents overlapping scheduled jobs.
- Tests cover schema validation and model serialization.

## Replacing demo data with real production data

In a real project, remove the `Generate demo inputs` step from `batch-inference.yml` and replace it with a step that reads data from your approved source: object storage, a database export, an API, or a generated CSV.

Keep the contract expected by `src/leadtime_ml/features.py`, or update the feature schema intentionally and retrain the model.
