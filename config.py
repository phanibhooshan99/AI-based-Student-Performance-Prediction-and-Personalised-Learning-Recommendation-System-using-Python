"""Central configuration: paths, feature lists and thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATASET_PATH = DATA_DIR / "students.csv"
RESOURCES_PATH = DATA_DIR / "resources.csv"

TOPICS = ["mathematics", "programming", "data_structures", "databases", "statistics"]
TOPIC_COLS = [f"topic_{t}" for t in TOPICS]

NUMERIC_FEATURES = [
    "study_hours", "attendance", "previous_score", "assignment_score",
    "quiz_score", "participation", "sleep_hours",
] + TOPIC_COLS
CATEGORICAL_FEATURES = ["gender", "internet_access", "extra_activities", "parental_support"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_REG = "final_score"
TARGET_CLS = "performance_level"

# final_score thresholds for the performance level
AT_RISK_BELOW = 50
HIGH_FROM = 75
LEVELS = ["At-Risk", "Average", "High"]

RANDOM_STATE = 42
