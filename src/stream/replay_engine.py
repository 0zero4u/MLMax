from collections import deque
import numpy as np
from tqdm import tqdm
from src.features.feature_calculators import calculate_features
from src.labels.triple_barrier import find_triple_barrier_label

RAW_WINDOW = 100

def build_dataset_from_unified_df(df, pt:float=0.0015, sl:float=0.0015, lookahead:int=200, trigger_on_book:bool=True):
    # state
    cvd = 0.0
    cvd_history = deque(maxlen=300)
    price_history = deque(maxlen=RAW_WINDOW)
    spread_history = deque(maxlen=RAW_WINDOW)
    raw_history = {"price": deque(maxlen=RAW_WINDOW), "volume": deque(maxlen=RAW_WINDOW), "spread": deque(maxlen=RAW_WINDOW)}

    human_features = []
    raw_sequences = []
    labels = []

    # iterate
    for i, row in tqdm(df.iterrows(), total=len(df)):
        stype = row["stream_type"]
        # update state
        if stype == "trade":
            sign = -1 if row["is_buyer_maker"] else 1
            quote = float(row["quote_qty"]) if (row.get("quote_qty") is not None and not np.isnan(row["quote_qty"])) else 0.0
            cvd += sign * quote
            cvd_history.append(cvd)
            price = float(row["price"]) if (row.get("price") is not None and not np.isnan(row["price"])) else None
            if price is not None:
                price_history.append(price)
                raw_history["price"].append(price)
                raw_history["volume"].append(quote)
        else: # book
            bid = float(row["best_bid_price"]) if not np.isnan(row.get("best_bid_price") if row.get("best_bid_price") is not None else np.nan) else None
            ask = float(row["best_ask_price"]) if not np.isnan(row.get("best_ask_price") if row.get("best_ask_price") is not None else np.nan) else None
            if bid is not None and ask is not None:
                spread = ask - bid
                spread_history.append(spread)
                raw_history["spread"].append(spread)

        # trigger
        if (trigger_on_book and stype == "book") or (not trigger_on_book and i % 1 == 0):
            state = {
                "cvd_history": cvd_history, "price_history": price_history, "spread_history": spread_history,
                "bid_qty": row.get("best_bid_qty"), "ask_qty": row.get("best_ask_qty"),
                "mid_price": row.get("mid_price"), "time": row.get("time")
            }
            features = calculate_features(state)
            
            if len(raw_history["price"]) == RAW_WINDOW:
                human_features.append(features)
                seq = np.stack([
                    np.array(raw_history["price"], dtype=np.float32),
                    np.array(raw_history["volume"], dtype=np.float32),
                    np.array(raw_history["spread"], dtype=np.float32)
                ], axis=1)
                raw_sequences.append(seq)
                label, ret, t_hit = find_triple_barrier_label(df, i, pt, sl, lookahead)
                labels.append({"time": int(row["time"]), "label": int(label), "ret": float(ret), "time_to_hit": int(t_hit)})

    # Align data by ensuring we only keep samples where all three components were generated
    min_len = min(len(human_features), len(raw_sequences), len(labels))
    human_features = human_features[:min_len]
    raw_sequences = np.array(raw_sequences[:min_len], dtype=np.float32)
    labels = labels[:min_len]

    return human_features, raw_sequences, labels
    
