"""Prediction + recommendation pipeline for a single student."""
import joblib
import pandas as pd

from .config import FEATURES, MODEL_DIR
from .recommender import (behaviour_tips, recommend_resources, study_plan,
                          weak_topics)
from .train import train_all

_cache = {}


def _load():
    if not _cache:
        reg_p, cls_p = MODEL_DIR / "score_regressor.joblib", MODEL_DIR / "level_classifier.joblib"
        if not (reg_p.exists() and cls_p.exists()):
            train_all(verbose=False)
        _cache["reg"], _cache["cls"] = joblib.load(reg_p), joblib.load(cls_p)
    return _cache["reg"], _cache["cls"]


def analyse_student(student: dict) -> dict:
    """Predict final score and level, then build a personalised plan."""
    reg, cls = _load()
    row = pd.DataFrame([student])[FEATURES]
    score = float(max(0, min(100, reg.predict(row)[0])))
    level = str(cls.predict(row)[0])
    proba = dict(zip(cls.classes_, cls.predict_proba(row)[0].round(3).tolist()))
    weak = weak_topics(student)
    return {
        "predicted_score": round(score, 1),
        "predicted_level": level,
        "level_probabilities": proba,
        "weak_topics": weak,
        "tips": behaviour_tips(student),
        "resources": recommend_resources(student),
        "study_plan": study_plan(level, weak),
    }


SAMPLE_STUDENT = {
    "gender": "Female", "study_hours": 1.5, "attendance": 68, "previous_score": 52,
    "assignment_score": 55, "quiz_score": 48, "participation": 3, "sleep_hours": 5.5,
    "internet_access": "Yes", "extra_activities": "No", "parental_support": "Medium",
    "topic_mathematics": 38, "topic_programming": 55, "topic_data_structures": 42,
    "topic_databases": 62, "topic_statistics": 45,
}
