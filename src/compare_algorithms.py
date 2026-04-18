"""CLI entrypoint to compare multiple training algorithms side by side."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn.model_selection as sk_model_selection

from ml_pipeline.model_factory import Algorithm
from ml_pipeline.training_service import TrainingConfig, TrainingService


def parse_args() -> argparse.Namespace:
    """Parse command-line options for multi-algorithm comparison."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--summary-out", default="reports/algorithm_comparison.json")
    parser.add_argument("--markdown-out", default="reports/algorithm_comparison.md")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--latency-runs", type=int, default=50)
    parser.add_argument(
        "--algorithms",
        default=",".join(Algorithm.choices()),
        help=(
            "Comma-separated algorithms to compare. "
            "Allowed: logistic_regression, random_forest, xgboost"
        ),
    )
    return parser.parse_args()


def parse_algorithms(raw_value: str) -> list[str]:
    """Validate and normalize requested algorithm names."""

    requested = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not requested:
        raise ValueError("No algorithms provided. Use --algorithms with at least one value.")

    allowed = set(Algorithm.choices())
    invalid = [value for value in requested if value not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported algorithms: {invalid}. Use one of: {Algorithm.choices()}"
        )

    return requested


def _load_model_and_artifact(model_path: str) -> tuple[Any, dict[str, Any]]:
    """Load a model and optional dict artifact metadata from disk."""

    raw_artifact = joblib.load(model_path)
    if isinstance(raw_artifact, dict) and "model" in raw_artifact:
        return raw_artifact["model"], raw_artifact

    return raw_artifact, {}


def append_efficiency_metrics(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Append model size and prediction latency for tie-breaking decisions."""

    if args.latency_runs <= 0:
        raise ValueError("--latency-runs must be greater than zero.")

    dataframe = pd.read_csv(args.data)
    if dataframe.empty:
        raise ValueError("Comparison dataset is empty. Provide at least one row.")

    for row in rows:
        if row.get("status") != "ok":
            continue

        model_path = row["model_out"]
        try:
            model, artifact = _load_model_and_artifact(model_path)
            target_name = artifact.get("target_name", "weakest_finger")
            if target_name not in dataframe.columns:
                raise ValueError(
                    f"Target column '{target_name}' not found in dataset '{args.data}'."
                )

            feature_names = artifact.get("feature_names")
            if not feature_names:
                feature_names = [column for column in dataframe.columns if column != target_name]

            feature_frame = dataframe[feature_names]
            target_series = dataframe[target_name]

            split_result = sk_model_selection.train_test_split(
                feature_frame,
                target_series,
                test_size=0.2,
                random_state=args.random_state,
                stratify=target_series,
            )
            _, x_test, _, _ = tuple(split_result)

            model.predict(x_test)  # warmup
            started_at = time.perf_counter()
            for _ in range(args.latency_runs):
                model.predict(x_test)
            ended_at = time.perf_counter()

            row["model_size_bytes"] = int(os.path.getsize(model_path))
            row["predict_ms_per_call"] = ((ended_at - started_at) / args.latency_runs) * 1000
            row["test_rows"] = int(len(x_test))
            row["latency_runs"] = int(args.latency_runs)
        except Exception as ex:
            row["efficiency_error"] = str(ex)


def _choose_best_by_efficiency(ranked_ok: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose fastest/smallest model among top-quality ties."""

    if not ranked_ok:
        return None

    top = ranked_ok[0]
    top_quality_group = [
        row
        for row in ranked_ok
        if row["accuracy"] == top["accuracy"]
        and row["macro_f1"] == top["macro_f1"]
        and row["weighted_f1"] == top["weighted_f1"]
        and row["support"] == top["support"]
    ]

    comparable = [
        row
        for row in top_quality_group
        if "predict_ms_per_call" in row and "model_size_bytes" in row
    ]
    if not comparable:
        return top

    return sorted(
        comparable,
        key=lambda row: (row["predict_ms_per_call"], row["model_size_bytes"]),
    )[0]


def write_markdown(summary: dict[str, Any], markdown_path: Path) -> None:
    """Write a human-readable markdown comparison report."""

    ranked_ok = summary.get("ranked_ok", [])
    failed_rows = [row for row in summary.get("rows", []) if row.get("status") != "ok"]

    lines: list[str] = []
    lines.append("# Algorithm Comparison")
    lines.append("")
    lines.append(f"- Data path: {summary.get('data_path')}")
    lines.append(f"- Random state: {summary.get('random_state')}")
    lines.append(f"- Algorithms evaluated: {len(summary.get('rows', []))}")
    lines.append("")
    lines.append("## Quality Metrics")
    lines.append("")
    lines.append("| Algorithm | Status | Accuracy | Macro F1 | Weighted F1 | Support |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in ranked_ok:
        lines.append(
            f"| {row['algorithm']} | {row['status']} | {row['accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | {row['support']} |"
        )

    if failed_rows:
        lines.append("")
        lines.append("## Failed Algorithms")
        lines.append("")
        lines.append("| Algorithm | Error |")
        lines.append("|---|---|")
        for row in failed_rows:
            lines.append(f"| {row['algorithm']} | {row.get('error', 'unknown error')} |")

    lines.append("")
    lines.append("## Efficiency Tie-Breakers")
    lines.append("")
    lines.append("| Algorithm | Model Size (bytes) | Predict ms/call | Test Rows |")
    lines.append("|---|---:|---:|---:|")
    for row in ranked_ok:
        model_size = row.get("model_size_bytes")
        predict_ms = row.get("predict_ms_per_call")
        test_rows = row.get("test_rows")
        if model_size is None or predict_ms is None:
            lines.append(f"| {row['algorithm']} | - | - | - |")
            continue
        lines.append(
            f"| {row['algorithm']} | {int(model_size)} | {float(predict_ms):.4f} | {int(test_rows)} |"
        )

    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    quality_tie_count = int(summary.get("quality_tie_count", 0))
    best_quality = summary.get("best")
    best_efficiency = summary.get("best_by_efficiency")

    if best_quality is None:
        lines.append("No successful algorithm run is available.")
    else:
        lines.append(
            f"- Best by quality metrics: {best_quality['algorithm']} "
            f"(accuracy={best_quality['accuracy']:.4f}, macro_f1={best_quality['macro_f1']:.4f})"
        )
        if quality_tie_count > 1 and best_efficiency is not None:
            lines.append(
                f"- Quality tie count: {quality_tie_count}. "
                f"Best practical choice by speed and size: {best_efficiency['algorithm']}."
            )

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_algorithms(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate all requested algorithms under identical settings."""

    service = TrainingService.default(random_state=args.random_state)
    algorithms = parse_algorithms(args.algorithms)

    rows: list[dict[str, Any]] = []
    for algorithm in algorithms:
        model_out = str(Path(args.model_dir) / f"model_compare_{algorithm}.joblib")
        report_out = str(Path(args.report_dir) / f"training_report_compare_{algorithm}.json")

        config = TrainingConfig(
            data_path=args.data,
            model_out=model_out,
            report_out=report_out,
            random_state=args.random_state,
            algorithm=algorithm,
        )

        try:
            report = service.train(config)
            rows.append(
                {
                    "algorithm": algorithm,
                    "status": "ok",
                    "accuracy": float(report.get("accuracy", 0.0)),
                    "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
                    "weighted_f1": float(
                        report.get("weighted avg", {}).get("f1-score", 0.0)
                    ),
                    "support": int(
                        round(float(report.get("macro avg", {}).get("support", 0.0)))
                    ),
                    "model_out": model_out,
                    "report_out": report_out,
                }
            )
        except Exception as ex:
            rows.append(
                {
                    "algorithm": algorithm,
                    "status": "failed",
                    "error": str(ex),
                    "model_out": model_out,
                    "report_out": report_out,
                }
            )

    ranked_ok = sorted(
        [row for row in rows if row.get("status") == "ok"],
        key=lambda row: (
            row["accuracy"],
            row["macro_f1"],
            row["weighted_f1"],
            row["support"],
        ),
        reverse=True,
    )

    append_efficiency_metrics(ranked_ok, args)
    best_by_efficiency = _choose_best_by_efficiency(ranked_ok)

    quality_tie_count = 0
    if ranked_ok:
        top = ranked_ok[0]
        quality_tie_count = sum(
            1
            for row in ranked_ok
            if row["accuracy"] == top["accuracy"]
            and row["macro_f1"] == top["macro_f1"]
            and row["weighted_f1"] == top["weighted_f1"]
            and row["support"] == top["support"]
        )

    return {
        "data_path": args.data,
        "random_state": args.random_state,
        "rows": rows,
        "ranked_ok": ranked_ok,
        "best": ranked_ok[0] if ranked_ok else None,
        "best_by_efficiency": best_by_efficiency,
        "quality_tie_count": quality_tie_count,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact, readable comparison summary."""

    print(
        "algorithm\tstatus\taccuracy\tmacro_f1\tweighted_f1\tsupport\tmodel_size\tpredict_ms\tmodel_out\treport_out"
    )

    ranked_ok = summary.get("ranked_ok", [])
    for row in ranked_ok:
        model_size = row.get("model_size_bytes")
        predict_ms = row.get("predict_ms_per_call")
        model_size_text = str(model_size) if model_size is not None else "-"
        predict_ms_text = f"{float(predict_ms):.4f}" if predict_ms is not None else "-"
        print(
            f"{row['algorithm']}\t{row['status']}\t{row['accuracy']:.4f}\t"
            f"{row['macro_f1']:.4f}\t{row['weighted_f1']:.4f}\t{row['support']}\t"
            f"{model_size_text}\t{predict_ms_text}\t"
            f"{row['model_out']}\t{row['report_out']}"
        )

    failed_rows = [row for row in summary.get("rows", []) if row.get("status") != "ok"]
    for row in failed_rows:
        print(
            f"{row['algorithm']}\t{row['status']}\t-\t-\t-\t-\t"
            f"{row['model_out']}\t{row['report_out']}"
        )
        print(f"ERROR[{row['algorithm']}]: {row.get('error', 'unknown error')}")

    best = summary.get("best")
    if best:
        print(f"\nBEST_BY_QUALITY={best['algorithm']}")

    best_efficiency = summary.get("best_by_efficiency")
    if best_efficiency:
        print(f"BEST_BY_EFFICIENCY={best_efficiency['algorithm']}")


def main() -> None:
    """Run comparison and persist machine-readable summary output."""

    args = parse_args()
    summary = compare_algorithms(args)
    print_summary(summary)

    output_path = Path(args.summary_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SUMMARY_JSON={output_path.as_posix()}")

    markdown_path = Path(args.markdown_out)
    write_markdown(summary, markdown_path)
    print(f"SUMMARY_MARKDOWN={markdown_path.as_posix()}")


if __name__ == "__main__":
    main()
