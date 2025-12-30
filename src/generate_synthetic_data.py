import pandas as pd, numpy as np, uuid, datetime, os

np.random.seed(7)

fingers = [
    "left_pinky","left_ring","left_middle","left_index",
    "right_index","right_middle","right_ring","right_pinky"
]

n_users = 4
sessions_per_user = 20  # 80 rows total
start = datetime.datetime(2025, 12, 5, 9, 0, 0)

weak_list = fingers * 10
np.random.shuffle(weak_list)

rows = []
idx = 0

for u in range(1, n_users + 1):
    user_id = f"u{u:02d}"
    base_wpm = np.random.uniform(30, 70)
    base_acc = np.random.uniform(0.88, 0.97)

    for s in range(sessions_per_user):
        weak = weak_list[idx]; idx += 1
        session_id = str(uuid.uuid4())
        ts = start + datetime.timedelta(days=s, minutes=int(np.random.randint(0, 180)))

        wpm = float(np.clip(base_wpm + 0.35*s + np.random.normal(0, 4), 15, 140))
        accuracy = float(np.clip(base_acc + 0.0009*s + np.random.normal(0, 0.012), 0.70, 0.995))

        errors = {}
        base_level = (1 - accuracy)
        for f in fingers:
            err = base_level*np.random.uniform(0.6, 1.1) + np.random.normal(0, 0.004)
            if f == weak:
                err += np.random.uniform(0.04, 0.10)
            errors[f] = float(np.clip(err, 0, 0.25))

        mx = max(errors.values())
        if errors[weak] < mx:
            errors[weak] = float(np.clip(mx + 0.01, 0, 0.25))

        row = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": ts.isoformat(timespec="seconds"),
            "wpm": round(wpm, 1),
            "accuracy": round(accuracy, 3),
        }

        for f in fingers:
            row[f"error_{f}"] = round(errors[f], 3)

        row["weakest_finger"] = weak
        rows.append(row)

df = pd.DataFrame(rows)

out_path = "data/raw/sessions.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)

print(f"✅ Wrote {len(df)} rows to {out_path}")
print(df["weakest_finger"].value_counts())