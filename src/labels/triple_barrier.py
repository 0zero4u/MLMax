from typing import Tuple
import numpy as np
import pandas as pd

def find_triple_barrier_label(df: pd.DataFrame, i: int, pt_pct: float, sl_pct: float, horizon_seconds: float, fee_pct: float) -> Tuple[int, float, int]:
    """
    Finds the label for an event at index `i` by looking ahead in the DataFrame
    for a fixed time duration.
    """
    entry_event = df.iloc[i]
    entry_time_ms = entry_event["time"]
    entry_ask = entry_event["best_ask_price"]
    entry_bid = entry_event["best_bid_price"]

    if np.isnan(entry_ask) or np.isnan(entry_bid):
        # Using horizon_seconds as a proxy for max duration in rows for timeout
        # A more precise row count could be found, but this is a safe upper bound
        return 0, 0.0, int(horizon_seconds * 100) # Approx max rows for timeout

    # Define barriers based on executable prices
    pt_long_price = entry_ask * (1 + pt_pct)
    sl_long_price = entry_ask * (1 - sl_pct)
    
    pt_short_price = entry_bid * (1 - pt_pct)
    sl_short_price = entry_bid * (1 + sl_pct)

    # --- MODIFIED: Time-based lookahead ---
    timeout_timestamp_ms = entry_time_ms + (horizon_seconds * 1000)
    
    # Create a mask for future events within the time horizon
    future_mask = (df.index > i) & (df['time'] <= timeout_timestamp_ms)
    future_events = df.loc[future_mask]
    
    # If no future events in horizon, it's a timeout
    if future_events.empty:
        return 0, 0.0, df.index[-1] - i # return remaining rows as time_to_hit

    for j, future_event in enumerate(future_events.itertuples()):
        time_to_hit = j + 1
        future_ask = future_event.best_ask_price
        future_bid = future_event.best_bid_price

        if np.isnan(future_ask) or np.isnan(future_bid):
            continue

        # --- Check Long Scenario ---
        if future_bid >= pt_long_price:
            realized_ret = (future_bid - entry_ask) / entry_ask
            net_ret = realized_ret - 2 * fee_pct
            return 1, net_ret, time_to_hit
        if future_bid <= sl_long_price:
            realized_ret = (future_bid - entry_ask) / entry_ask
            net_ret = realized_ret - 2 * fee_pct
            return 2, net_ret, time_to_hit

        # --- Check Short Scenario ---
        if future_ask <= pt_short_price:
            realized_ret = (entry_bid - future_ask) / entry_bid
            net_ret = realized_ret - 2 * fee_pct
            return -1, net_ret, time_to_hit
        if future_ask >= sl_short_price:
            realized_ret = (entry_bid - future_ask) / entry_bid
            net_ret = realized_ret - 2 * fee_pct
            return 2, net_ret, time_to_hit

    # If loop completes, it's a timeout
    return 0, 0.0, len(future_events)
