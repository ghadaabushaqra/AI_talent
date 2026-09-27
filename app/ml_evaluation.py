"""Reproducible evaluation for the separate assignment-success ML model.

The supplied resume dataset has no human assignment-success labels.  The
generated outcomes are therefore suitable for an engineering benchmark only;
they must never be presented as proof of real-world hiring accuracy.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "documented_skill_count",
    "documented_experience_years",
    "documented_project_count",
    "documented_education_count",
    "documented_certification_count",
]


def success_features(profile: dict[str, Any]) -> list[float]:
    """Return only source-documented CV features; unknown values stay neutral."""
    return [
        float(len(profile.get("skills") or [])),
        float(profile.get("experience_years") or 0),
        float(len(profile.get("projects") or profile.get("past_projects") or [])),
        float(len(profile.get("education") or [])),
        float(len(profile.get("certifications") or [])),
    ]


def labelled_matrix(candidates: list[dict[str, Any]], outcomes: list[dict[str, Any]]):
    labels = {row["candidate_id"]: int(row["assignment_success"]) for row in outcomes}
    labelled = [row for row in candidates if row["id"] in labels]
    return [success_features(row) for row in labelled], [labels[row["id"]] for row in labelled], labelled


def train_success_model(candidates: list[dict[str, Any]], outcomes: list[dict[str, Any]]):
    X, y, _ = labelled_matrix(candidates, outcomes)
    base = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=500, random_state=7))
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5).fit(X, y)


def _weighted_baseline(features: list[list[float]]) -> list[int]:
    """Independent, documented-feature baseline used only for comparison."""
    predictions = []
    for skills, years, projects, education, certifications in features:
        score = .60 * min(1.0, skills / 6) + .25 * min(1.0, years / 8) + .15 * min(1.0, projects)
        predictions.append(int(score >= .52))
    return predictions


def _metrics(y_true: list[int], predictions: list[int], probabilities: list[float] | None = None) -> dict[str, Any]:
    result = {
        "accuracy": round(accuracy_score(y_true, predictions), 3),
        "precision": round(precision_score(y_true, predictions, zero_division=0), 3),
        "recall": round(recall_score(y_true, predictions, zero_division=0), 3),
        "f1": round(f1_score(y_true, predictions, zero_division=0), 3),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }
    if probabilities is not None and len(set(y_true)) == 2:
        result["roc_auc"] = round(roc_auc_score(y_true, probabilities), 3)
        result["brier_score"] = round(brier_score_loss(y_true, probabilities), 3)
    return result


def evaluate_success_model(candidates: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    X, y, labelled = labelled_matrix(candidates, outcomes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=7, stratify=y)
    base = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=500, random_state=7))
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5).fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1].tolist()
    predictions = [int(value >= .5) for value in probabilities]
    baseline = _weighted_baseline(X_test)
    balance = Counter(y)
    return {
        "label_provenance": {
            "type": "synthetic_proxy",
            "human_ground_truth": False,
            "warning": "Assignment outcomes were generated from documented CV signals and deterministic noise; metrics measure pipeline behavior, not real hiring validity.",
        },
        "dataset": {
            "labelled_profiles": len(labelled),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "class_balance": {"successful": balance[1], "not_successful": balance[0]},
            "split": "75/25 stratified, random_state=7",
        },
        "features": FEATURE_NAMES,
        "classifier": {"model": "Sigmoid-calibrated StandardScaler + LogisticRegression", **_metrics(y_test, predictions, probabilities)},
        "deterministic_baseline": {"model": "documented-signal threshold", **_metrics(y_test, baseline)},
        "matching_evaluation": {
            "status": "pending_human_review",
            "note": "Precision@K/NDCG for candidate ranking requires job-specific relevance judgements from an HR reviewer; no labels are fabricated.",
        },
        "leakage_review": "Candidate ID, name, source category, availability, and weighted-match score are excluded from ML features.",
    }
