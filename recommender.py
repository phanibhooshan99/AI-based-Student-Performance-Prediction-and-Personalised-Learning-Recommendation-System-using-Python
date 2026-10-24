"""Rule + content based personalised learning recommender."""
import pandas as pd

from .config import RESOURCES_PATH, TOPICS

WEAK_BELOW = 60


def difficulty_for(score: float) -> str:
    if score < 40:
        return "beginner"
    if score < 65:
        return "intermediate"
    return "advanced"


def weak_topics(student: dict, threshold: float = WEAK_BELOW) -> list:
    """Return [(topic, score)] sorted from weakest to strongest below the threshold."""
    scores = [(t, float(student[f"topic_{t}"])) for t in TOPICS]
    return sorted([s for s in scores if s[1] < threshold], key=lambda x: x[1])


def behaviour_tips(student: dict) -> list:
    tips = []
    if student["attendance"] < 75:
        tips.append("Raise attendance above 75% - missed classes are a strong predictor of lower scores.")
    if student["study_hours"] < 2:
        tips.append("Increase focused self-study to at least 2-3 hours per day, using short Pomodoro sessions.")
    if student["sleep_hours"] < 6:
        tips.append("Aim for 7-8 hours of sleep; memory consolidation drops sharply below 6 hours.")
    elif student["sleep_hours"] > 9:
        tips.append("Very long sleep can reduce study time - try a consistent 7-8 hour schedule.")
    if student["participation"] < 4:
        tips.append("Take part in class discussions or study groups to reinforce understanding.")
    if student["assignment_score"] < 60:
        tips.append("Submit assignments on time and review teacher feedback on every one.")
    if student["internet_access"] == "No":
        tips.append("Use offline resources or library/lab hours to access online learning material.")
    if not tips:
        tips.append("Great habits! Keep your routine and move on to advanced practice.")
    return tips


def recommend_resources(student: dict, top_n: int = 6, resources: pd.DataFrame = None) -> pd.DataFrame:
    """Recommend learning resources for the weakest topics at a matching difficulty."""
    if resources is None:
        resources = pd.read_csv(RESOURCES_PATH)
    weak = weak_topics(student)
    if not weak:  # strong student -> challenge them on their lowest topics
        scores = sorted([(t, float(student[f"topic_{t}"])) for t in TOPICS], key=lambda x: x[1])
        weak = scores[:2]

    picks = []
    for topic, score in weak:
        diff = difficulty_for(score)
        subset = resources[(resources["topic"] == topic) & (resources["difficulty"] == diff)]
        if subset.empty:
            subset = resources[resources["topic"] == topic]
        subset = subset.assign(reason=f"{topic.replace('_', ' ').title()} score {score:.0f} -> {diff} level")
        picks.append(subset)
    out = pd.concat(picks).drop_duplicates("resource_id")
    # interleave topics so weakest topics come first but every weak topic is represented
    out = out.groupby("topic", sort=False).head(2)
    return out.head(top_n).reset_index(drop=True)


def study_plan(level: str, weak: list) -> list:
    focus = ", ".join(t.replace("_", " ") for t, _ in weak[:3]) or "revision and advanced practice"
    if level == "At-Risk":
        return [f"Mon-Fri: 45 min on weakest topics ({focus})",
                "Sat: take a mock quiz and review mistakes",
                "Sun: 1 mentor / teacher doubt-clearing session"]
    if level == "Average":
        return [f"Mon-Thu: 30 min on {focus}",
                "Fri: practice problems / mini project",
                "Sat: timed quiz, Sun: rest + light revision"]
    return [f"Mon-Wed: 30 min advanced practice on {focus}",
            "Thu-Fri: project or competitive problems",
            "Weekend: help peers (teaching strengthens mastery)"]
