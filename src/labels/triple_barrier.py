# src/labels/triple_barrier.py
"""
Triple barrier labeler.
Simple, clear implementation that looks forward `max_horizon` rows from index i
and returns label in {1, -1, 0}.
It uses mid_price when available, otherwise trade price.
Fees are applied (fee_pct is round-trip percentage).
"""

from typing import Tuple
import numpy as np

def find_triple_barrier_label(df, i:int, pt:float, sl:float, max_horizon:int, fee_pct:float=0.00075):
    """
    df: unified dataframe (must have 'mid_price' or 'price' columns)
    i: current row index (entry)
    pt, sl: profit-take / stop-loss as relative fractions (e.g., 0.0015)
    max_horizon: number of rows to look ahead
    fee_pct: per trade fee fraction (apply both entry+exit -> approx 2*fee_pct)
    Returns: label (1, -1, 0)
    """
    n = len(df)
    entry = df.iloc[i]
    # entry price: prefer mid_price if present else price
    entry_price = entry.get("mid_price") if not np.isnan(entry.get("mid_price") if entry.get("mid_price") is not None else np.nan) else entry.get("price")
    if entry_price is None or np.isnan(entry_price):
        return 0

    # apply fees: to be conservative, require profit beyond fees
    round_trip_fee = 2 * fee_pct
    upper = entry_price * (1.0 + pt + round_trip_fee)
    lower = entry_price * (1.0 - sl - round_trip_fee)

    # iterate forward
    end = min(n, i + max_horizon + 1)
    for j in range(i+1, end):
        r = df.iloc[j]
        # use future executable price: mid_price if present else trade price
        p = r.get("mid_price") if not np.isnan(r.get("mid_price") if r.get("mid_price") is not None else np.nan) else r.get("price")
        if p is None or np.isnan(p):
            continue
        if p >= upper:
            return 1
        if p <= lower:
            return -1
    # timeout
    return 0
  
