"""CLI entrypoint for training weakest-finger classification models."""

import argparse

from ml_pipeline.model_factory import Algorithm
from ml_pipeline.training_service import TrainingConfig, TrainingService


def parse_args() -> argparse.Namespace:
    """Parse command-line options for training runs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--report-out", default="reports/training_report.json")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--algorithm",
        default=Algorithm.LOGISTIC_REGRESSION.value,
        choices=Algorithm.choices(),
        help="Algorithm to train: logistic_regression, random_forest, or xgboost.",
    )
    return parser.parse_args()


def main() -> None:
    """Build training config and delegate execution to TrainingService."""

    args = parse_args()
    service = TrainingService.default(random_state=args.random_state)
    config = TrainingConfig(
        data_path=args.data,
        model_out=args.model_out,
        report_out=args.report_out,
        random_state=args.random_state,
        algorithm=args.algorithm,
    )
    service.train(config)


if __name__ == "__main__":
    main()
