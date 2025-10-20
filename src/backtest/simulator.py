import pandas as pd
import numpy as np
from tqdm import tqdm

def simulate_trades(df_unified: pd.DataFrame, 
                    predictions_df: pd.DataFrame, 
                    pt_pct: float, 
                    sl_pct: float, 
                    max_horizon: int, 
                    stake: float = 1.0, 
                    fee_pct: float = 0.00075):
    """
    Simulates trades based on model predictions, mirroring the triple-barrier labeling logic.

    Args:
        df_unified: The unified, time-sorted DataFrame of all market events.
        predictions_df: DataFrame with 'time' and 'prediction' columns (1 for long, -1 for short).
        pt_pct: Profit-take percentage (e.g., 0.001 for 0.1%), must match labeling.
        sl_pct: Stop-loss percentage (e.g., 0.001 for 0.1%), must match labeling.
        max_horizon: Maximum number of rows to hold a trade, must match labeling.
        stake: The amount of capital to use for each trade.
        fee_pct: Taker fee percentage for entry and exit.

    Returns:
        A tuple of (summary_stats_dict, trade_log_df).
    """
    df = df_unified.reset_index(drop=True)
    predictions = predictions_df.copy()
    
    # Ensure we can quickly look up index by time
    time_to_idx = pd.Series(df.index, index=df['time'])

    trade_log = []

    for _, row in tqdm(predictions.iterrows(), total=len(predictions), desc="Backtesting Trades"):
        pred_time = row["time"]
        pred_label = int(row["prediction"])

        if pred_label == 0:  # No trade signal
            continue

        try:
            i = time_to_idx[pred_time]
        except KeyError:
            # Prediction timestamp not found in the main dataframe
            continue

        entry_event = df.iloc[i]
        entry_ask = entry_event["best_ask_price"]
        entry_bid = entry_event["best_bid_price"]

        if np.isnan(entry_ask) or np.isnan(entry_bid):
            continue

        exit_reason = "timeout"
        exit_price = np.nan
        exit_time = -1
        
        # --- Define Barriers Based on Prediction ---
        if pred_label == 1: # Long Trade
            entry_price = entry_ask
            pt_price = entry_price * (1 + pt_pct)
            sl_price = entry_price * (1 - sl_pct)
        elif pred_label == -1: # Short Trade
            entry_price = entry_bid
            pt_price = entry_price * (1 - pt_pct)
            sl_price = entry_price * (1 + sl_pct)
        else:
            continue

        # --- Look Ahead for Barrier Hit ---
        future_events = df.iloc[i + 1 : i + 1 + max_horizon]
        for j, future_event in enumerate(future_events.itertuples()):
            future_ask = future_event.best_ask_price
            future_bid = future_event.best_bid_price

            if np.isnan(future_ask) or np.isnan(future_bid):
                continue

            if pred_label == 1: # Long Exit Logic
                if future_bid >= pt_price:
                    exit_price = future_bid
                    exit_reason = "profit_take"
                    break
                if future_bid <= sl_price:
                    exit_price = future_bid
                    exit_reason = "stop_loss"
                    break
            
            elif pred_label == -1: # Short Exit Logic
                if future_ask <= pt_price:
                    exit_price = future_ask
                    exit_reason = "profit_take"
                    break
                if future_ask >= sl_price:
                    exit_price = future_ask
                    exit_reason = "stop_loss"
                    break
        
        # If loop finished, it was a timeout
        if exit_reason == "timeout":
            last_event = future_events.iloc[-1]
            if pred_label == 1: # Closing a long
                exit_price = last_event["best_bid_price"]
            else: # Closing a short
                exit_price = last_event["best_ask_price"]
        
        if np.isnan(exit_price):
            continue # Could not determine a valid exit price

        # --- Calculate PnL ---
        if pred_label == 1: # Long PnL
            ret = (exit_price - entry_price) / entry_price
        else: # Short PnL
            ret = (entry_price - exit_price) / entry_price
        
        net_ret = ret - (2 * fee_pct) # Fees for entry and exit
        pnl = stake * net_ret
        
        trade_log.append({
            "entry_time": pred_time,
            "exit_time": df.loc[i + j + 1, "time"] if exit_reason != "timeout" else future_events.iloc[-1]["time"],
            "prediction": pred_label,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pt_level": pt_price,
            "sl_level": sl_price,
            "gross_ret": ret,
            "net_ret": net_ret,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "duration_rows": j + 1 if exit_reason != "timeout" else max_horizon
        })

    if not trade_log:
        return {"trades": 0}, pd.DataFrame()

    log_df = pd.DataFrame(trade_log)
    
    # --- Generate Summary Statistics ---
    total_trades = len(log_df)
    win_trades = log_df[log_df["pnl"] > 0]
    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
    
    summary = {
        "total_trades": total_trades,
        "total_pnl": log_df["pnl"].sum(),
        "mean_pnl": log_df["pnl"].mean(),
        "win_rate": win_rate,
        "profit_factor": log_df[log_df["pnl"] > 0]["pnl"].sum() / abs(log_df[log_df["pnl"] < 0]["pnl"].sum()),
        "sharpe_ratio": (log_df["net_ret"].mean() / log_df["net_ret"].std()) * np.sqrt(252 * 24 * 60) # Annualized Sharpe (approx)
    }

    return summary, log_df
