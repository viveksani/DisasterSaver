"""
DisasterSaver — Flood Prediction Pipeline (Person 1)
Module: ml/flood_model.py
Description: ChennaiFloodModel — Random Forest wrapper for urban flood risk prediction.

Design:
- Wraps sklearn RandomForestClassifier with a project-specific interface.
- predict_proba() returns class-1 (flooded) probability as a 1D array.
- enforce_physics_monotonicity() applies a bounded, deterministic post-processing boost
  when the drainage overload ratio R_overload > 1. This is a heuristic rule, not part
  of the Random Forest training objective.
- Risk thresholds (LOW / MODERATE / HIGH) are project-defined engineering cutoffs, not
  statistically calibrated probability thresholds.

Target Environment: Google Colab (/content/ml/flood_model.py) / Python 3.8+
"""

import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier

# Feature names must remain in this exact order — they define column indexing
# throughout the entire pipeline (preprocessing → training → inference).
FEATURE_NAMES = [
    "elevation",        # Terrain elevation (m)
    "slope",            # Surface slope (degrees)
    "tpi",              # Topographic Position Index (m)
    "dist_to_drain",    # Distance to nearest drain geometry (m)
    "q_cap",            # Manning channel conveyance capacity Q_cap (m³/s)
    "q_demand",         # Rational Method runoff demand Q_demand (m³/s)
    "r_overload",       # Drainage overload ratio R_overload = Q_demand / Q_cap
]

# Project-defined risk thresholds (engineering cutoffs, not statistically calibrated).
RISK_LOW_THRESHOLD = 0.33
RISK_HIGH_THRESHOLD = 0.66


class ChennaiFloodModel:
    """
    Random Forest–based urban flood risk classifier for Chennai, India.

    Wraps sklearn RandomForestClassifier with:
    - A deterministic, bounded physics monotonicity post-processing rule.
    - Named feature importance reporting.
    - Explicit API contracts expected by train.py and predict.py.

    Parameters
    ----------
    n_estimators : int
        Number of decision trees (default 100).
    max_depth : int
        Maximum tree depth (default 12).
    random_state : int
        Random seed for reproducibility (default 42).
    class_weight : str or dict
        Class weighting strategy (default "balanced").
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
        class_weight: str = "balanced",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.feature_names = FEATURE_NAMES

        self.estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChennaiFloodModel":
        """
        Train the Random Forest classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 7)
            Feature matrix. Columns must match FEATURE_NAMES order.
        y : np.ndarray, shape (n_samples,)
            Binary flood labels: 1 = flooded, 0 = dry.

        Returns
        -------
        self
        """
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features, got {X.shape[1]}."
            )
        self.estimator.fit(X, y)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class-1 (flooded) probability as a 1D float32 array.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 7)

        Returns
        -------
        probs : np.ndarray, shape (n_samples,), dtype float32
            Values in [0.0, 1.0].
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.estimator.predict_proba(X)[:, 1].astype(np.float32)

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Return mean impurity-decrease feature importances keyed by feature name.

        Returns
        -------
        Dict mapping each name in FEATURE_NAMES to its importance value.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, self.estimator.feature_importances_)
        }

    # ------------------------------------------------------------------
    # Physics post-processing
    # ------------------------------------------------------------------

    def enforce_physics_monotonicity(
        self,
        scores: np.ndarray,
        r_overload: np.ndarray,
        boost_factor: float = 0.15,
    ) -> np.ndarray:
        """
        Heuristic post-processing rule: increase risk scores when the drainage
        overload ratio R_overload > 1 (demand exceeds channel capacity).

        This is NOT part of Random Forest training. It is a deterministic
        domain-knowledge correction applied after model inference to enforce the
        physically intuitive relationship: higher drainage stress → higher (or equal)
        flood risk.

        Formula
        -------
        For R_overload <= 1  : adjustment = 0
        For R_overload > 1   : adjustment = min(boost_factor, (R_overload - 1) × 0.05)
        adjusted_score       = clip(score + adjustment, 0.0, 1.0)

        Properties guaranteed:
        - adjustment >= 0 for all inputs (scores never decreased by this rule)
        - adjustment <= boost_factor (bounded)
        - adjusted scores remain in [0.0, 1.0]
        - output is finite (no NaN or Inf)

        Parameters
        ----------
        scores : np.ndarray, shape (n,), dtype float32
            Raw model risk probabilities in [0, 1].
        r_overload : np.ndarray, shape (n,)
            Drainage overload ratio (Q_demand / Q_cap) per cell.
        boost_factor : float
            Maximum allowed upward adjustment (default 0.15).

        Returns
        -------
        adjusted : np.ndarray, shape (n,), dtype float32
            Physics-adjusted risk scores in [0.0, 1.0].
        """
        scores = np.asarray(scores, dtype=np.float32)
        r_overload = np.asarray(r_overload, dtype=np.float64)

        # Bounded linear boost for overloaded cells only
        excess = np.maximum(0.0, r_overload - 1.0)
        adjustment = np.minimum(boost_factor, excess * 0.05).astype(np.float32)

        adjusted = np.clip(scores + adjustment, 0.0, 1.0)

        # Safety guard — should never trigger under normal inputs
        if np.isnan(adjusted).any() or np.isinf(adjusted).any():
            raise RuntimeError(
                "enforce_physics_monotonicity produced NaN/Inf values. "
                "Check input scores and r_overload arrays."
            )

        return adjusted

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_risk(adjusted_scores: np.ndarray) -> np.ndarray:
        """
        Map continuous adjusted risk scores to categorical risk classes.

        Thresholds are project-defined engineering cutoffs:
            LOW      : adjusted_score < 0.33
            MODERATE : 0.33 <= adjusted_score < 0.66
            HIGH     : adjusted_score >= 0.66

        Parameters
        ----------
        adjusted_scores : np.ndarray, shape (n,)

        Returns
        -------
        risk_classes : np.ndarray of str, shape (n,)
        """
        risk_classes = np.full(adjusted_scores.shape, "LOW", dtype=object)
        risk_classes[
            (adjusted_scores >= RISK_LOW_THRESHOLD) & (adjusted_scores < RISK_HIGH_THRESHOLD)
        ] = "MODERATE"
        risk_classes[adjusted_scores >= RISK_HIGH_THRESHOLD] = "HIGH"
        return risk_classes

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"ChennaiFloodModel({status}, n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, random_state={self.random_state})"
        )
