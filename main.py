"""Command line interface.

Usage:
    python main.py generate     # create data/students.csv
    python main.py train        # train + evaluate models
    python main.py demo         # predict + recommend for a sample student
"""
import argparse

from src.data_generator import ensure_dataset
from src.predict import SAMPLE_STUDENT, analyse_student
from src.train import train_all


def show(result: dict) -> None:
    print(f"\nPredicted final score : {result['predicted_score']}")
    print(f"Predicted level       : {result['predicted_level']}  {result['level_probabilities']}")
    print("\nWeak topics:")
    for t, s in result["weak_topics"]:
        print(f"  - {t.replace('_', ' ').title()}: {s:.0f}")
    print("\nHabit tips:")
    for t in result["tips"]:
        print(f"  * {t}")
    print("\nRecommended resources:")
    for _, r in result["resources"].iterrows():
        print(f"  - [{r['difficulty']}] {r['title']} ({r['type']}) - {r['url']}")
    print("\nStudy plan:")
    for line in result["study_plan"]:
        print(f"  > {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Student performance AI")
    parser.add_argument("command", choices=["generate", "train", "demo"])
    args = parser.parse_args()
    if args.command == "generate":
        df = ensure_dataset(force=True)
        print(f"Generated {len(df)} students.")
    elif args.command == "train":
        train_all()
    else:
        show(analyse_student(SAMPLE_STUDENT))


if __name__ == "__main__":
    main()
