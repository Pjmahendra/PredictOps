"""Model definitions and hyperparameter search spaces for supervised failure detection.

Every model is wrapped in a Pipeline(preprocessor -> estimator) so the scaler/imputer
is re-fitted on each training fold only (no leakage) and is saved together with the model.

Why each algorithm (short version — full explanations in docs/05_ml_models.md):
  logreg  linear baseline; coefficients are directly interpretable
  tree    single decision tree; human-readable rules, shows the "easy split"
  rf      bagged trees; robust to noise, less variance than one tree
  svm     RBF kernel SVM; non-linear boundary with a max-margin principle
  xgboost gradient-boosted trees; usually strongest on tabular data, native SHAP support
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from predictops.preprocessing.pipeline import make_preprocessor

SEED = 42

# Small, computationally sensible search spaces (sampled with ParameterSampler)
SEARCH_SPACES = {
    "logreg": {"C": [0.001, 0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]},
    "tree": {"max_depth": [2, 3, 4, 6, 8, None], "min_samples_leaf": [1, 20, 100, 500]},
    "rf": {"n_estimators": [100, 200], "max_depth": [4, 8, 12, None], "min_samples_leaf": [1, 10, 50],
           "max_features": ["sqrt", 0.3]},
    "svm": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
    "xgboost": {"n_estimators": [100, 300, 600], "max_depth": [2, 3, 4, 6], "learning_rate": [0.03, 0.1, 0.3],
                "subsample": [0.7, 1.0], "colsample_bytree": [0.5, 0.8, 1.0], "min_child_weight": [1, 5, 20]},
}
N_ITER = {"logreg": 5, "tree": 12, "rf": 10, "svm": 6, "xgboost": 15}

# SVM training cost grows ~quadratically with rows, so it is trained on a stratified
# subsample (all positives + random negatives) — documented in docs/05.
SVM_MAX_ROWS = 20_000


def make_estimator(name: str, params: dict, pos_weight: float):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced", **params)
    if name == "tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **params)
    if name == "rf":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **params)
    if name == "svm":
        return SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED, **params)
    if name == "xgboost":
        return XGBClassifier(scale_pos_weight=pos_weight, n_jobs=8, random_state=SEED,
                             eval_metric="aucpr", tree_method="hist", **params)
    raise ValueError(name)


def make_pipeline(name: str, params: dict, features: list[str], pos_weight: float) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(features)), ("model", make_estimator(name, params, pos_weight))])
