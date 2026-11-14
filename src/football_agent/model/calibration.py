"""Logistic calibration layer.

A multinomial logistic regression maps raw model outputs
(P(home), P(draw), P(away), Elo win prob, form difference) to calibrated
outcome probabilities. Fit on out-of-sample backtest predictions; if there
is too little data it returns the raw probabilities unchanged.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class Calibrator:
    def __init__(self) -> None:
        self.model: LogisticRegression | None = None

    def fit(self, features: np.ndarray, outcomes: np.ndarray) -> "Calibrator":
        """features: (n, 5) rows; outcomes: 0=home win, 1=draw, 2=away win."""
        if len(outcomes) < 30 or len(set(outcomes.tolist())) < 3:
            self.model = None
            return self
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf.fit(features, outcomes)
        self.model = clf
        return self

    def calibrate(self, features: np.ndarray) -> np.ndarray | None:
        if self.model is None:
            return None
        return self.model.predict_proba(features.reshape(1, -1))[0]  # type: ignore[no-any-return]
