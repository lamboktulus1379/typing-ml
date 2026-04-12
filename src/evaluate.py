"""CLI entrypoint for evaluating trained weakest-finger models."""

import argparse

from ml_pipeline.evaluation_service import EvaluationConfig, EvaluationService


def parse_args() -> argparse.Namespace:
    """Parse command-line options for evaluation runs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Build evaluation config and delegate execution to EvaluationService."""

    args = parse_args()
    service = EvaluationService.default()
    config = EvaluationConfig(
        data_path=args.data,
        model_path=args.model,
        fig_dir=args.fig_dir,
        random_state=args.random_state,
    )
    service.evaluate(config)


if __name__ == "__main__":
    main()
