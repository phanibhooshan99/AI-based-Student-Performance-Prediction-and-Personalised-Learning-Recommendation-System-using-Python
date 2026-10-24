import pandas as pd
import pytest

from src.config import FEATURES, TOPIC_COLS
from src.data_generator import generate_students
from src.predict import SAMPLE_STUDENT, analyse_student
from src.recommender import (behaviour_tips, difficulty_for,
                             recommend_resources, weak_topics)


def test_dataset_shape_and_columns():
    df = generate_students(200)
    assert len(df) == 200
    for col in FEATURES + ["final_score", "performance_level"]:
        assert col in df.columns
    assert df["final_score"].between(0, 100).all()
    assert set(df["performance_level"]) <= {"At-Risk", "Average", "High"}


def test_weak_topics_sorted():
    weak = weak_topics(SAMPLE_STUDENT)
    scores = [s for _, s in weak]
    assert scores == sorted(scores)
    assert all(s < 60 for s in scores)


@pytest.mark.parametrize("score,expected", [(20, "beginner"), (50, "intermediate"), (90, "advanced")])
def test_difficulty(score, expected):
    assert difficulty_for(score) == expected


def test_recommendations_match_weak_topics():
    recs = recommend_resources(SAMPLE_STUDENT)
    weak = {t for t, _ in weak_topics(SAMPLE_STUDENT)}
    assert len(recs) > 0
    assert set(recs["topic"]) <= weak


def test_tips_for_poor_habits():
    assert len(behaviour_tips(SAMPLE_STUDENT)) >= 3


def test_full_pipeline():
    result = analyse_student(SAMPLE_STUDENT)
    assert 0 <= result["predicted_score"] <= 100
    assert result["predicted_level"] in {"At-Risk", "Average", "High"}
    assert isinstance(result["resources"], pd.DataFrame)
    assert result["study_plan"]
