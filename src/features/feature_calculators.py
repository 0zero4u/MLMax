"""
Feature extraction functions. `calculate_features` is the main entry point.
State object must contain all necessary deques and WelfordStats objects.
"""

import numpy as np
from collections import deque
from typing import Dict, Any

EPS = 1e-9

def calculate_features(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates features based on the current state.
    This version uses efficient WelfordStats for normalization.
    """
    features = {}

    # --- Order Book & Price Features ---
    bid_p = state.get("best_bid_price")
    ask_p = state.get("best_ask_price")
    bid_q = state.get("best_bid_qty")
    ask_q = state.get("best_ask_qty")

    if bid_p is None or ask_p is None or bid_q is None or ask_q is None:
        return {} # Not enough data

    # Order book imbalance
    total_qty = bid_q + ask_q
    features["obi"] = (bid_q - ask_q) / (total_qty + EPS)

    # Spread
    spread = ask_p - bid_p
    mid_price = (ask_p + bid_p) / 2
    rel_spread = spread / (mid_price + EPS)
    state["welford_spread"].update(rel_spread)
    features["spread_norm"] = state["welford_spread"].normalize(rel_spread)
    
    # Microprice and Log-Return RV (Original)
    microprice = (bid_p * ask_q + ask_p * bid_q) / (total_qty + EPS)
    state["price_history"].append(microprice)
    
    if len(state["price_history"]) >= 20: # Ensure enough data for RV
        price_arr = np.array(state["price_history"])
        log_returns = np.log(price_arr[1:] / (price_arr[:-1] + EPS))
        rv = np.sum(log_returns**2)
        state["welford_rv"].update(rv)
        features["rv_norm"] = state["welford_rv"].normalize(rv)
    else:
        features["rv_norm"] = 0.0

    # --- NEW: Tactical Realized Volatility (from Live Script) ---
    if len(state["mid_price_history"]) >= 20:
        mid_price_arr = np.array(state["mid_price_history"])
        # Variance of mid-price changes is a simple measure of jitter
        rv_tactical = np.var(np.diff(mid_price_arr))
        state["welford_rv_tactical"].update(rv_tactical)
        features["rv_tactical_norm"] = state["welford_rv_tactical"].normalize(rv_tactical)
    else:
        features["rv_tactical_norm"] = 0.0

    # --- CVD Features (Time-Aware) ---
    now = state["time"] / 1000.0 # Convert ms to seconds
    cvd_history = state.get("cvd_history", deque())

    def get_windowed_cvd(window_sec):
        target_time = now - window_sec
        # Find the CVD value at the start of the window
        start_cvd = 0
        for ts, val in reversed(cvd_history):
            if ts < target_time:
                start_cvd = val
                break
        current_cvd = cvd_history[-1][1] if cvd_history else 0
        return current_cvd - start_cvd

    cvd_5s = get_windowed_cvd(5)
    state["welford_cvd_5s"].update(cvd_5s)
    features["cvd_5s_zscore"] = state["welford_cvd_5s"].normalize(cvd_5s)

    cvd_1m = get_windowed_cvd(60)
    state["welford_cvd_1m"].update(cvd_1m)
    features["cvd_1m_zscore"] = state["welford_cvd_1m"].normalize(cvd_1m)

    cvd_3m = get_windowed_cvd(180)
    state["welford_cvd_3m"].update(cvd_3m)
    features["cvd_3m_zscore"] = state["welford_cvd_3m"].normalize(cvd_3m)

    features["time"] = state.get("time")
    return features
    
