import numpy as np
import pandas as pd
import pytest

from pipeline.metrics import bootstrap_skill, coverage, nmae, nrmse, pinball, skill


def test_basic_errors():
    y = np.array([0.0, 1.0, 2.0])
    f = np.array([1.0, 1.0, 0.0])
    assert nmae(y, f) == pytest.approx(1.0)
    assert nrmse(y, f) == pytest.approx(np.sqrt(5 / 3))


def test_skill_is_zero_for_reference_and_one_for_perfect():
    y = np.array([0.2, 0.5, 0.9])
    ref = np.array([0.1, 0.7, 0.4])
    assert skill(y, ref, ref) == 0.0
    assert skill(y, y, ref) == 1.0


def test_pinball_matches_hand_value():
    y = np.array([1.0])
    q = np.array([[0.0, 1.0, 2.0]])
    # 0.1*(1-0) + 0 + (0.9-1)*(1-2)... = 0.1 + 0 + 0.1, mean of three
    assert pinball(y, q) == pytest.approx(0.2 / 3)


def test_coverage():
    y = np.array([0.0, 0.5, 1.0, 2.0])
    assert coverage(y, np.zeros(4), np.ones(4)) == 0.75


def test_bootstrap_interval_is_deterministic_and_contains_point():
    rng = np.random.default_rng(1)
    days = pd.Series(pd.date_range("2024-01-01", periods=120, freq="D").repeat(10))
    y = rng.random(len(days))
    ref = y + rng.normal(0, 0.3, len(y))
    f = y + rng.normal(0, 0.2, len(y))
    a = bootstrap_skill(y, f, ref, days, reps=300)
    b = bootstrap_skill(y, f, ref, days, reps=300)
    assert a == b
    assert a[0] < skill(y, f, ref) < a[1]
