"""CLI tool for generating synthetic typing-session datasets."""

import argparse
import datetime
import os
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.constants import FINGERS


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Immutable input configuration for synthetic data generation."""

    n_users: int
    sessions_per_user: int
    seed: int
    output_path: str


class SyntheticTypingDatasetGenerator:
    """Generates realistic tabular samples for weakest-finger modeling."""

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate CLI/config constraints before generation starts."""

        if self.config.n_users <= 0:
            raise ValueError(f"--n-users must be > 0, got {self.config.n_users}")
        if self.config.sessions_per_user <= 0:
            raise ValueError(
                f"--sessions-per-user must be > 0, got {self.config.sessions_per_user}"
            )
        if not str(self.config.output_path).strip():
            raise ValueError("--output must be a non-empty path")

    def generate_dataframe(self) -> pd.DataFrame:
        """Generate synthetic sessions and return a ready-to-save DataFrame."""

        rng = np.random.default_rng(self.config.seed)
        start = datetime.datetime(2025, 12, 1, 9, 0, 0)
        rows: list[dict[str, Any]] = []

        total_rows = self.config.n_users * self.config.sessions_per_user
        print(f"Generating {total_rows:,} rows of synthetic keystroke data...")

        for user_index in range(1, self.config.n_users + 1):
            user_id = f"u{user_index:04d}"
            base_wpm = rng.uniform(28, 78)
            base_acc = rng.uniform(0.86, 0.97)
            base_dwell = rng.uniform(72, 105)
            base_flight = rng.uniform(165, 260)

            weak_finger = str(rng.choice(FINGERS))

            for session_index in range(self.config.sessions_per_user):
                session_id = str(uuid.uuid4())
                timestamp = start + datetime.timedelta(
                    days=session_index,
                    minutes=int(rng.integers(0, 240)),
                )

                wpm = float(np.clip(base_wpm + 0.55 * session_index + rng.normal(0, 3.2), 15, 145))
                accuracy = float(
                    np.clip(base_acc + 0.0012 * session_index + rng.normal(0, 0.009), 0.68, 0.997)
                )

                if rng.random() < 0.18:
                    weak_finger = str(rng.choice(FINGERS))

                row: dict[str, Any] = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": timestamp.isoformat(timespec="seconds"),
                    "wpm": round(wpm, 1),
                    "accuracy": round(accuracy, 3),
                    "weakest_finger": weak_finger,
                }

                base_error_level = max(0.0, 1.0 - accuracy)

                for finger in FINGERS:
                    error_value = base_error_level * rng.uniform(0.65, 1.05) + rng.normal(0, 0.002)
                    dwell_value = base_dwell + rng.normal(0, 5.5)
                    flight_value = (base_flight - (wpm * 0.45)) + rng.normal(0, 11)

                    if finger == weak_finger:
                        error_value += rng.uniform(0.05, 0.115)
                        dwell_value += rng.uniform(28, 58)
                        flight_value += rng.uniform(65, 120)

                    row[f"error_{finger}"] = round(float(np.clip(error_value, 0, 0.30)), 3)
                    row[f"dwell_{finger}"] = round(float(np.clip(dwell_value, 40, 300)), 1)
                    row[f"flight_{finger}"] = round(float(np.clip(flight_value, 50, 500)), 1)

                rows.append(row)

        return pd.DataFrame(rows)

    def save(self, dataframe: pd.DataFrame) -> None:
        """Persist generated rows as CSV, creating parent folders when needed."""

        output_dir = os.path.dirname(self.config.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        dataframe.to_csv(self.config.output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for synthetic data generation."""

    parser = argparse.ArgumentParser(description="Generate synthetic typing dataset.")
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--sessions-per-user", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/dataset.csv")
    return parser.parse_args()


def main() -> None:
    """Build config, generate dataset, save CSV, and print summary."""

    args = parse_args()
    config = SyntheticDataConfig(
        n_users=args.n_users,
        sessions_per_user=args.sessions_per_user,
        seed=args.seed,
        output_path=args.output,
    )

    generator = SyntheticTypingDatasetGenerator(config)
    dataframe = generator.generate_dataframe()
    generator.save(dataframe)

    print(f"Success! Saved to {config.output_path}")
    print(
        f"Rows: {len(dataframe):,} | Users: {config.n_users} | "
        f"Sessions/user: {config.sessions_per_user}"
    )


if __name__ == "__main__":
    main()
