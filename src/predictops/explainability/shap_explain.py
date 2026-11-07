"""SHAP explanations for the supervised detector and the anomaly detector.

* XGBoost (inside a sklearn Pipeline): TreeExplainer on the TRANSFORMED features — exact,
  fast (milliseconds per row), so it is used live by the API.
  Output is in log-odds units: base value + sum(contributions) = model logit.
* One-Class SVM anomaly detector: model-agnostic KernelExplainer on its decision score
  with a small background sample — slow, so used offline / for a few selected minutes.

SHAP tells us how much each feature pushed THIS MODEL's output up or down for THIS
input. It is not causal: a high contribution means "the model relied on it", not
"this sensor caused the failure".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


class TreeShapExplainer:
    """Wraps a fitted Pipeline(pre -> XGBClassifier)."""

    def __init__(self, pipeline):
        self.pre = pipeline.named_steps["pre"]
        self.model = pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        Z = self.pre.transform(X)
        ex = self.explainer(Z)
        ex.data = X[Z.columns].to_numpy()          # show ORIGINAL (unscaled) values in plots
        ex.feature_names = list(Z.columns)
        return ex

    def contributions(self, X: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """JSON-friendly per-row explanation used by the API/dashboard."""
        ex = self.explain(X)
        out = []
        for i in range(len(X)):
            v = ex.values[i]
            order = np.argsort(-np.abs(v))[:top_k]
            out.append({"base_value_logit": float(np.atleast_1d(ex.base_values)[i] if np.ndim(ex.base_values) else ex.base_values),
                        "contributions": [{"feature": ex.feature_names[j], "value": float(ex.data[i, j]),
                                           "shap": float(v[j])} for j in order]})
        return out


class AnomalyShapExplainer:
    """KernelSHAP on an AnomalyDetector's score (higher = more anomalous)."""

    def __init__(self, detector, background: pd.DataFrame, n_background: int = 50, seed: int = 42):
        self.det = detector
        bg = background[detector.features].sample(min(n_background, len(background)), random_state=seed)
        self.feature_names = detector.features
        f = lambda A: detector.score(pd.DataFrame(A, columns=detector.features))
        self.explainer = shap.KernelExplainer(f, bg.to_numpy())

    def explain(self, X: pd.DataFrame, nsamples: int = 300) -> shap.Explanation:
        vals = self.explainer.shap_values(X[self.feature_names].to_numpy(), nsamples=nsamples, silent=True)
        return shap.Explanation(values=vals, base_values=np.full(len(X), self.explainer.expected_value),
                                data=X[self.feature_names].to_numpy(), feature_names=self.feature_names)
