import argparse
import datetime
import os
import uuid
from typing import Any

import numpy as np
import pandas as pd


FINGERS = [
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]


def generate_dataset(
    n_users: int,
    sessions_per_user: int,
    seed: int,
) -> pd.DataFrame:
    if n_users <= 0:
        raise ValueError(f"n_users must be > 0, got {n_users}")
    if sessions_per_user <= 0:
        raise ValueError(
            f"sessions_per_user must be > 0, got {sessions_per_user}"
        )

    rng = np.random.default_rng(seed)
    start = datetime.datetime(2025, 12, 1, 9, 0, 0)
    rows: list[dict[str, Any]] = []

    print(f"Generating {n_users * sessions_per_user:,} rows of synthetic keystroke data...")

    for u in range(1, n_users + 1):
        user_id = f"u{u:04d}"
        base_wpm = rng.uniform(28, 78)
        base_acc = rng.uniform(0.86, 0.97)
        base_dwell = rng.uniform(72, 105)
        base_flight = rng.uniform(165, 260)

        weak_finger = str(rng.choice(FINGERS))

        for s in range(sessions_per_user):
            session_id = str(uuid.uuid4())
            ts = start + datetime.timedelta(days=s, minutes=int(rng.integers(0, 240)))

            # Mild learning trend: users generally improve over sessions.
            wpm = float(np.clip(base_wpm + 0.55 * s + rng.normal(0, 3.2), 15, 145))
            accuracy = float(np.clip(base_acc + 0.0012 * s + rng.normal(0, 0.009), 0.68, 0.997))

            if rng.random() < 0.18:
                weak_finger = str(rng.choice(FINGERS))

            row: dict[str, Any] = {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": ts.isoformat(timespec="seconds"),
                "wpm": round(wpm, 1),
                "accuracy": round(accuracy, 3),
                "weakest_finger": weak_finger,
            }

            base_error_level = max(0.0, 1.0 - accuracy)

            for finger in FINGERS:
                err = base_error_level * rng.uniform(0.65, 1.05) + rng.normal(0, 0.002)
                dwell = base_dwell + rng.normal(0, 5.5)
                flight = (base_flight - (wpm * 0.45)) + rng.normal(0, 11)

                if finger == weak_finger:
                    err += rng.uniform(0.05, 0.115)
                    dwell += rng.uniform(28, 58)
                    flight += rng.uniform(65, 120)

                row[f"error_{finger}"] = round(float(np.clip(err, 0, 0.30)), 3)
                row[f"dwell_{finger}"] = round(float(np.clip(dwell, 40, 300)), 1)
                row[f"flight_{finger}"] = round(float(np.clip(flight, 50, 500)), 1)

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic typing dataset.")
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--sessions-per-user", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/dataset.csv")
    args = parser.parse_args()

    if args.n_users <= 0:
        raise ValueError(f"--n-users must be > 0, got {args.n_users}")
    if args.sessions_per_user <= 0:
        raise ValueError(
            f"--sessions-per-user must be > 0, got {args.sessions_per_user}"
        )
    if not str(args.output).strip():
        raise ValueError("--output must be a non-empty path")

    df = generate_dataset(args.n_users, args.sessions_per_user, args.seed)
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Success! Saved to {output_path}")
    print(f"Rows: {len(df):,} | Users: {args.n_users} | Sessions/user: {args.sessions_per_user}")


if __name__ == "__main__":
    main()
