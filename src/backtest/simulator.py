import pandas as pd
import numpy as np
from tqdm import tqdm

def simulate_trades(df_unified: pd.DataFrame, 
                    predictions_df: pd.DataFrame, 
                    pt_pct: float, 
                    sl_pct: float, 
                    horizon_seconds: float, # MODIFIED: from max_horizon to horizon_seconds
                    stake: float = 1.0, 
                    fee_pct: float = 0.00075):
    """
    Simulates trades based on model predictions, using a time-based horizon.
    """
    df = df_unified.reset_index(drop=True)
    predictions = predictions_df.copy()
    
    time_to_idx = pd.Series(df.index, index=df['time'])
    trade_log = []

    for _, row in tqdm(predictions.iterrows(), total=len(predictions), desc="Backtesting Trades"):
        pred_time = row["time"]
        pred_label = int(row["prediction"])

        if pred_label == 0: continue

        try: i = time_to_idx[pred_time]
        except KeyError: continue

        entry_event = df.iloc[i]
        entry_ask = entry_event["best_ask_price"]
        entry_bid = entry_event["best_bid_price"]

        if np.isnan(entry_ask) or np.isnan(entry_bid): continue

        exit_reason = "timeout"
        exit_price = np.nan
        exit_time = -1
        
        if pred_label == 1: # Long
            entry_price = entry_ask
            pt_price = entry_price * (1 + pt_pct)
            sl_price = entry_price * (1 - sl_pct)
        elif pred_label == -1: # Short
            entry_price = entry_bid
            pt_price = entry_price * (1 - pt_pct)
            sl_price = entry_price * (1 + sl_pct)
        else: continue

        # --- MODIFIED: Look Ahead with Time-based Barrier ---
        timeout_timestamp_ms = pred_time + (horizon_seconds * 1000)
        future_mask = (df.index > i) & (df['time'] <= timeout_timestamp_ms)
        future_events = df.loc[future_mask]

        last_valid_event_idx = -1

        for j, future_event in enumerate(future_events.itertuples()):
            future_ask = future_event.best_ask_price
            future_bid = future_event.best_bid_price
            last_valid_event_idx = future_event.Index # Store the pandas Index

            if np.isnan(future_ask) or np.isnan(future_bid): continue

            if pred_label == 1: # Long Exit
                if future_bid >= pt_price: exit_reason = "profit_take"; exit_price = future_bid; break
                if future_bid <= sl_price: exit_reason = "stop_loss"; exit_price = future_bid; break
            elif pred_label == -1: # Short Exit
                if future_ask <= pt_price: exit_reason = "profit_take"; exit_price = future_ask; break
                if future_ask >= sl_price: exit_reason = "stop_loss"; exit_price = future_ask; break
        
        if exit_reason == "timeout":
            if not future_events.empty:
                last_event = future_events.iloc[-1]
                exit_price = last_event["best_bid_price"] if pred_label == 1 else last_event["best_ask_price"]
                last_valid_event_idx = last_event.name
        
        if np.isnan(exit_price): continue

        # ... (PnL calculation remains the same) ...
        if pred_label == 1: ret = (exit_price - entry_price) / entry_price
        else: ret = (entry_price - exit_price) / entry_price
        
        net_ret = ret - (2 * fee_pct)
        pnl = stake * net_ret
        
        trade_log.append({
            "entry_time": pred_time,
            "exit_time": df.loc[last_valid_event_idx, "time"],
            "prediction": pred_label, "entry_price": entry_price, "exit_price": exit_price,
            "pt_level": pt_price, "sl_level": sl_price, "gross_ret": ret, "net_ret": net_ret,
            "pnl": pnl, "exit_reason": exit_reason,
            "duration_rows": last_valid_event_idx - i
        })

    if not trade_log: return {"trades": 0}, pd.DataFrame()

    log_df = pd.DataFrame(trade_log)
    # ... (Summary stats calculation remains the same) ...
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
