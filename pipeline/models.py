"""Forecasters. Each takes a training table and a test table and returns predictions
as a fraction of the AC rating (`y_n` units). Nothing here sees test targets."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.config import QUANTILES, SEED
from pipeline.features import FEATURES

GBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def trainable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["daytime"] & df["y_n"].notna() & df["inputs_ok"]]


def persistence(test: pd.DataFrame) -> np.ndarray:
    return test["y_prev_n"].to_numpy()


def smart_persistence(test: pd.DataFrame) -> np.ndarray:
    return (test["k_prev"] * test["p_cs_n"]).to_numpy()


def physical(test: pd.DataFrame) -> np.ndarray:
    return test["p_phys_n"].to_numpy()


def physical_calibrated(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Least-squares scale and offset on the physical forecast (the linear sanity check)."""
    t = trainable(train)
    a, b = np.polyfit(t["p_phys_n"].to_numpy(), t["y_n"].to_numpy(), 1)
    return a * test["p_phys_n"].to_numpy() + b


def gbm(train: pd.DataFrame, test: pd.DataFrame, objective: str = "regression",
        alpha: float | None = None) -> np.ndarray:
    t = trainable(train)
    params = dict(GBM_PARAMS, objective=objective)
    if alpha is not None:
        params["alpha"] = alpha
    m = lgb.LGBMRegressor(**params)
    m.fit(t[list(FEATURES)], t["y_n"])
    return m.predict(test[list(FEATURES)])


def gbm_quantiles(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Columns are the QUANTILES, sorted per row so the bands never cross."""
    cols = [gbm(train, test, "quantile", q) for q in QUANTILES]
    return np.sort(np.column_stack(cols), axis=1)


def finish(pred: np.ndarray, test: pd.DataFrame, cap_n: float) -> np.ndarray:
    """Night is zero; nothing below zero or above the inverters' limit."""
    out = np.clip(pred, 0.0, cap_n)
    night = ~test["daytime"].to_numpy()
    out[night] = 0.0
    return out
