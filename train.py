"""Model training: compares models for score regression and performance-level classification."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (CATEGORICAL_FEATURES, FEATURES, MODEL_DIR, NUMERIC_FEATURES,
                     RANDOM_STATE, REPORT_DIR, TARGET_CLS, TARGET_REG)
from .data_generator import ensure_dataset


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def _pipe(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def train_all(verbose: bool = True) -> dict:
    df = ensure_dataset()
    X = df[FEATURES]
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, df[TARGET_REG], df[TARGET_CLS], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df[TARGET_CLS])

    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    metrics = {"regression": {}, "classification": {}}
    best_reg, best_reg_r2 = None, -1e9
    fitted_reg = {}
    for name, model in regressors.items():
        p = _pipe(model).fit(X_tr, yr_tr)
        fitted_reg[name] = p
        pred = p.predict(X_te)
        r2, mae = r2_score(yr_te, pred), mean_absolute_error(yr_te, pred)
        metrics["regression"][name] = {"r2": round(r2, 4), "mae": round(mae, 3)}
        if r2 > best_reg_r2:
            best_reg, best_reg_r2, best_reg_name = p, r2, name

    best_cls, best_cls_f1 = None, -1.0
    for name, model in classifiers.items():
        p = _pipe(model).fit(X_tr, yc_tr)
        pred = p.predict(X_te)
        acc, f1 = accuracy_score(yc_te, pred), f1_score(yc_te, pred, average="macro")
        metrics["classification"][name] = {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}
        if f1 > best_cls_f1:
            best_cls, best_cls_f1, best_cls_name = p, f1, name

    metrics["best"] = {"regression": best_reg_name, "classification": best_cls_name}

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(best_reg, MODEL_DIR / "score_regressor.joblib")
    joblib.dump(best_cls, MODEL_DIR / "level_classifier.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # tree models expose feature importances; linear models do not
    _plot_importance(fitted_reg["GradientBoosting"])

    if verbose:
        print(json.dumps(metrics, indent=2))
    return metrics


def _plot_importance(pipeline: Pipeline) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    names = pipeline.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(model.feature_importances_, index=names).sort_values().tail(12)
    ax = imp.plot.barh(figsize=(8, 5), color="#4C78A8")
    ax.set_title("Top factors influencing final score")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    train_all()
