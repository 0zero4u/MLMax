from typing import Tuple
import numpy as np
import pandas as pd

def find_triple_barrier_label(df: pd.DataFrame, i: int, pt_pct: float, sl_pct: float, max_horizon: int, fee_pct: float) -> Tuple[int, float, int]:
    """
    Finds the label for an event at index `i` by looking ahead in the DataFrame.

    Args:
        df: The unified, time-sorted DataFrame of all market events.
        i: The current row index (the point of entry).
        pt_pct: Profit-take percentage (e.g., 0.001 for 0.1%).
        sl_pct: Stop-loss percentage (e.g., 0.001 for 0.1%).
        max_horizon: Maximum number of rows to look ahead for a barrier hit.
        fee_pct: Taker fee percentage (e.g., 0.0004 for 0.04%).

    Returns:
        A tuple of (label, realized_return, time_to_hit).
        label:
            1: Long win (profit take)
           -1: Short win (profit take)
            0: Timeout (time horizon)
            2: Loss (stop loss)
        realized_return: The net return after fees for the regressor.
        time_to_hit: Number of rows until a barrier was hit.
    """
    entry_event = df.iloc[i]
    entry_ask = entry_event["best_ask_price"]
    entry_bid = entry_event["best_bid_price"]

    if np.isnan(entry_ask) or np.isnan(entry_bid):
        return 0, 0.0, max_horizon

    # Define barriers based on executable prices
    pt_long_price = entry_ask * (1 + pt_pct)
    sl_long_price = entry_ask * (1 - sl_pct)
    
    pt_short_price = entry_bid * (1 - pt_pct)
    sl_short_price = entry_bid * (1 + sl_pct)

    # Look ahead for a barrier hit
    future_events = df.iloc[i + 1 : i + 1 + max_horizon]

    for j, future_event in enumerate(future_events.itertuples()):
        time_to_hit = j + 1
        future_ask = future_event.best_ask_price
        future_bid = future_event.best_bid_price

        if np.isnan(future_ask) or np.isnan(future_bid):
            continue

        # --- Check Long Scenario ---
        # Profit take: can sell at the future bid
        if future_bid >= pt_long_price:
            realized_ret = (future_bid - entry_ask) / entry_ask
            net_ret = realized_ret - 2 * fee_pct
            return 1, net_ret, time_to_hit
        # Stop loss: forced to sell at the future bid
        if future_bid <= sl_long_price:
            realized_ret = (future_bid - entry_ask) / entry_ask
            net_ret = realized_ret - 2 * fee_pct
            return 2, net_ret, time_to_hit # CHANGED: Loss is now label 2

        # --- Check Short Scenario ---
        # Profit take: can buy back at the future ask
        if future_ask <= pt_short_price:
            realized_ret = (entry_bid - future_ask) / entry_bid
            net_ret = realized_ret - 2 * fee_pct
            return -1, net_ret, time_to_hit
        # Stop loss: forced to buy back at the future ask
        if future_ask >= sl_short_price:
            realized_ret = (entry_bid - future_ask) / entry_bid
            net_ret = realized_ret - 2 * fee_pct
            return 2, net_ret, time_to_hit # CHANGED: Loss is now label 2

    # If no barrier was hit, it's a timeout with zero return
    return 0, 0.0, max_horizon
