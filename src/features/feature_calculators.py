# src/features/feature_calculators.py
"""
Feature extraction functions. calculate_features(state) returns a dict of features.
State must contain:
 - price_history (deque of floats)
 - spread_history (deque of floats)
 - cvd_history (deque of floats)
 - bid_qty, ask_qty, mid_price, time
"""

import numpy as np
from collections import deque
from typing import Dict, Any

EPS = 1e-9

def safe_mean_std(arr):
    if len(arr) == 0:
        return np.nan, np.nan
    a = np.asarray(arr, dtype=float)
    return a.mean(), a.std(ddof=0)

def calculate_features(state: Dict[str, Any]) -> Dict[str, float]:
    features = {}
    price_hist = state.get("price_history", deque())
    spread_hist = state.get("spread_history", deque())
    cvd_hist = state.get("cvd_history", deque())

    # Price statistics
    p_mean, p_std = safe_mean_std(price_hist)
    features["price_mean_100"] = p_mean
    features["price_std_100"] = p_std if not np.isnan(p_std) else 0.0
    features["rv_norm"] = np.log1p((p_std / (abs(p_mean) + EPS)) if p_mean else p_std)

    # Spread stats
    s_mean, s_std = safe_mean_std(spread_hist)
    features["spread_mean"] = s_mean
    features["spread_std"] = s_std

    # Order book imbalance
    bid = float(state.get("bid_qty") or 0.0)
    ask = float(state.get("ask_qty") or 0.0)
    tot = bid + ask
    features["obi"] = (bid - ask) / tot if tot > 0 else 0.0

    # CVD zscore
    cvd = np.asarray(cvd_hist, dtype=float)
    if len(cvd) >= 10:
        features["cvd_mean"] = float(cvd.mean())
        features["cvd_std"] = float(cvd.std(ddof=0) if cvd.std(ddof=0) > 0 else 1e-6)
        features["cvd_5s_zscore"] = float((cvd[-1] - features["cvd_mean"]) / features["cvd_std"])
    else:
        features["cvd_mean"] = np.nan
        features["cvd_std"] = np.nan
        features["cvd_5s_zscore"] = np.nan

    # micro features: momentum / returns
    if len(price_hist) >= 2:
        p0 = float(price_hist[-1])
        p1 = float(price_hist[-2])
        features["price_momentum"] = p0 - p1
        features["price_return"] = (p0 / (p1 + EPS)) - 1.0
    else:
        features["price_momentum"] = np.nan
        features["price_return"] = np.nan

    # mid price and spread normalized
    features["mid_price"] = float(state.get("mid_price") or np.nan)
    features["time"] = int(state.get("time") or 0)
    return features
  
