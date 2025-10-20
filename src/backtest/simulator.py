import pandas as pd
import numpy as np

def naive_trade_sim(df_unified, labels_df, stake=1.0, fee_pct=0.00075):
    """
    For every label=1 open a long at time, close when label outcome happened or time limit.
    This naive simulator assumes entry at mid_price and exit at first barrier event we used in labeling.
    Returns a small summary.
    """
    df = df_unified.reset_index(drop=True)
    labels = labels_df
    results = []
    for _, row in labels.iterrows():
        t = row["time"]
        lab = int(row["label"])
        # find index in df
        idx = df[df["time"]==t].index
        if len(idx)==0:
            continue
        i = idx[0]
        entry_price = df.loc[i, "mid_price"] if not np.isnan(df.loc[i, "mid_price"]) else df.loc[i,"price"]
        # naive exit: search forward for price change sign agreeing with label or timeout
        # simple heuristic here:
        exit_price = entry_price
        n = len(df)
        for j in range(i+1, min(n, i+500)):
            p = df.loc[j, "mid_price"] if not np.isnan(df.loc[j, "mid_price"]) else df.loc[j,"price"]
            if np.isnan(p):
                continue
            if lab == 1 and p > entry_price:
                exit_price = p
                break
            if lab == -1 and p < entry_price:
                exit_price = p
                break
        ret = (exit_price - entry_price) / entry_price
        ret_net = ret - 2*fee_pct
        pnl = stake * ret_net
        results.append(pnl)
    results = np.array(results)
    return {
        "trades": len(results),
        "mean_pnl": float(np.nanmean(results)) if len(results)>0 else 0.0,
        "sum_pnl": float(np.nansum(results)) if len(results)>0 else 0.0
        }
    
