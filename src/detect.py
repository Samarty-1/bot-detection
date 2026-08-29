"""Detectors, weakest to strongest.

The order matters. The rule baseline is what a site ships on day one; the
isolation forest is the honest unsupervised setting (no labels, just "what
looks unlike normal traffic"); the gradient booster is a cheating upper bound
that assumes labels you would not actually have.

Every model exposes the same interface: fit on a feature frame, return a score
in [0, 1] where higher means more bot-like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RuleBaseline:
    """Hand-written rules: datacenter IP, or a UA that contradicts the TLS stack.

    Catches the naive tier essentially for free and the advanced tier never,
    which is exactly why nobody stops here.
    """

    name = "rules (datacenter or UA/TLS mismatch)"

    def fit(self, X: pd.DataFrame) -> "RuleBaseline":
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        hit = (X["is_datacenter"] > 0.5) | (X["ua_tls_consistent"] < 0.5)
        return hit.to_numpy(dtype=float)


class IsolationForestDetector:
    """Unsupervised outlier detection -- the realistic production setting.

    Fitted on the full mixed stream because in reality you cannot pre-filter to
    clean traffic. `contamination` is your prior on the bot rate, and getting it
    wrong is one of the main ways this model disappoints.
    """

    name = "isolation forest (unsupervised)"

    def __init__(self, contamination: float = 0.02, seed: int = 7):
        self.contamination = contamination
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=400,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        self.model.fit(self.scaler.fit_transform(X))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        # score_samples: higher = more normal. Flip and min-max to [0, 1].
        raw = -self.model.score_samples(self.scaler.transform(X))
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / (hi - lo + 1e-12)


class SupervisedUpperBound:
    """Gradient boosting on ground-truth labels.

    Not deployable -- you do not have these labels on live traffic. It is here
    to answer "how much signal is in the features at all", which tells you
    whether a disappointing unsupervised result is a feature problem or a
    method problem.
    """

    name = "gradient boosting (labelled upper bound)"

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.model = GradientBoostingClassifier(random_state=seed)
        self._holdout: tuple | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupervisedUpperBound":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=self.seed
        )
        self.model.fit(X_tr, y_tr)
        self._holdout = (X_te, y_te)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def holdout(self) -> tuple:
        if self._holdout is None:
            raise RuntimeError("fit() first")
        return self._holdout

    def importances(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=X.columns).sort_values(ascending=False)
