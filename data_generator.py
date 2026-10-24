"""Synthetic student dataset generator (use your own CSV with the same columns for real data)."""
import numpy as np
import pandas as pd

from .config import (AT_RISK_BELOW, DATASET_PATH, HIGH_FROM, RANDOM_STATE,
                     TOPIC_COLS)


def level_from_score(score: float) -> str:
    if score < AT_RISK_BELOW:
        return "At-Risk"
    if score >= HIGH_FROM:
        return "High"
    return "Average"


def generate_students(n: int = 1500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)  # hidden latent ability

    study_hours = np.clip(3 + 0.8 * ability + rng.normal(0, 1.2, n), 0, 10)
    attendance = np.clip(80 + 6 * ability + rng.normal(0, 9, n), 35, 100)
    previous_score = np.clip(65 + 13 * ability + rng.normal(0, 6, n), 15, 100)
    assignment_score = np.clip(68 + 10 * ability + rng.normal(0, 8, n), 10, 100)
    quiz_score = np.clip(64 + 12 * ability + rng.normal(0, 8, n), 10, 100)
    participation = np.clip(5 + 1.5 * ability + rng.normal(0, 1.5, n), 0, 10)
    sleep_hours = np.clip(rng.normal(7, 1.2, n), 3, 10)

    data = {
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "gender": rng.choice(["Male", "Female"], n),
        "study_hours": study_hours.round(1),
        "attendance": attendance.round(1),
        "previous_score": previous_score.round(1),
        "assignment_score": assignment_score.round(1),
        "quiz_score": quiz_score.round(1),
        "participation": participation.round(1),
        "sleep_hours": sleep_hours.round(1),
        "internet_access": rng.choice(["Yes", "No"], n, p=[0.85, 0.15]),
        "extra_activities": rng.choice(["Yes", "No"], n, p=[0.55, 0.45]),
        "parental_support": rng.choice(["Low", "Medium", "High"], n, p=[0.2, 0.45, 0.35]),
    }
    for col in TOPIC_COLS:
        data[col] = np.clip(quiz_score + rng.normal(0, 14, n), 5, 100).round(1)
    df = pd.DataFrame(data)

    support_bonus = df["parental_support"].map({"Low": -2.0, "Medium": 0.0, "High": 2.0})
    net_bonus = np.where(df["internet_access"] == "Yes", 1.0, -1.0)
    sleep_pen = -1.5 * np.abs(df["sleep_hours"] - 7.5)
    final = (
        0.30 * df["previous_score"] + 0.15 * df["assignment_score"]
        + 0.15 * df["quiz_score"] + 0.10 * df[TOPIC_COLS].mean(axis=1)
        + 0.10 * df["attendance"] + 2.2 * df["study_hours"]
        + 0.8 * df["participation"] + support_bonus + net_bonus + sleep_pen
        + rng.normal(0, 3, n)
    ) - 2
    df["final_score"] = np.clip(final, 0, 100).round(1)
    df["performance_level"] = df["final_score"].apply(level_from_score)
    return df


def ensure_dataset(force: bool = False) -> pd.DataFrame:
    """Load the dataset, generating it first if it does not exist."""
    if force or not DATASET_PATH.exists():
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_students().to_csv(DATASET_PATH, index=False)
    return pd.read_csv(DATASET_PATH)


if __name__ == "__main__":
    df = ensure_dataset(force=True)
    print(f"Saved {len(df)} rows to {DATASET_PATH}")
    print(df["performance_level"].value_counts())
