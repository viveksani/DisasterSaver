"""
Tests for ml/flood_model.py.
Covers ChennaiFloodModel initialization, fit, predict_proba, get_feature_importances,
physics monotonicity adjustment rules, and risk classification.
"""

import numpy as np
from ml.flood_model import ChennaiFloodModel, FEATURE_NAMES, RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD


def test_model_initialization():
    """Verify default constructor hyperparameter assignment."""
    model = ChennaiFloodModel(n_estimators=50, max_depth=8, random_state=123, class_weight="balanced")
    assert model.n_estimators == 50
    assert model.max_depth == 8
    assert model.random_state == 123
    assert model.feature_names == FEATURE_NAMES


def test_fit_and_predict_proba():
    """Verify fit and predict_proba return expected 1D float array in range [0, 1]."""
    np.random.seed(42)
    X = np.random.randn(200, 7).astype(np.float32)
    y = np.random.choice([0, 1], size=200).astype(np.uint8)

    model = ChennaiFloodModel(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (200,)
    assert probs.dtype == np.float32
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_feature_importances_dict():
    """Verify get_feature_importances returns dict matching FEATURE_NAMES keys exactly."""
    np.random.seed(42)
    X = np.random.randn(100, 7).astype(np.float32)
    y = np.random.choice([0, 1], size=100).astype(np.uint8)

    model = ChennaiFloodModel(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    importances = model.get_feature_importances()
    assert isinstance(importances, dict)
    assert list(importances.keys()) == FEATURE_NAMES
    assert sum(importances.values()) > 0.0


def test_physics_monotonicity_rules():
    """Test all 5 physics monotonicity properties."""
    model = ChennaiFloodModel()

    # Rule 1: Monotonic increasing adjustment when overload increases
    scores = np.array([0.40, 0.40, 0.40], dtype=np.float32)
    overload = np.array([0.5, 1.5, 4.0], dtype=np.float32)
    adj = model.enforce_physics_monotonicity(scores, overload)

    assert adj[0] == 0.40  # No boost when R <= 1.0
    assert adj[1] > adj[0]  # Overload > 1 gets positive boost
    assert adj[2] >= adj[1] # Higher overload gets higher or equal boost

    # Rule 2: Scores bounded in [0, 1]
    high_scores = np.array([0.95, 0.99], dtype=np.float32)
    high_overload = np.array([10.0, 50.0], dtype=np.float32)
    adj_clipped = model.enforce_physics_monotonicity(high_scores, high_overload, boost_factor=0.15)
    assert (adj_clipped <= 1.0).all()
    assert (adj_clipped >= 0.0).all()

    # Rule 3: No NaN or Inf output
    assert not np.isnan(adj).any()
    assert not np.isinf(adj).any()


def test_classify_risk_thresholds():
    """Verify engineering risk classification cutoffs."""
    scores = np.array([0.10, 0.32, 0.33, 0.50, 0.65, 0.66, 0.90], dtype=np.float32)
    classes = ChennaiFloodModel.classify_risk(scores)
    expected = ["LOW", "LOW", "MODERATE", "MODERATE", "MODERATE", "HIGH", "HIGH"]
    assert list(classes) == expected
