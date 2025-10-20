import pandas as pd
import numpy as np
from collections import deque
from tqdm import tqdm
import argparse
import os

from src.ingest.loader import load_and_merge
from src.features.welford import WelfordStats
from src.features.feature_calculators import calculate_features
from src.labels.triple_barrier import find_triple_barrier_label
from src.utils.io import save_parquet, save_npy

# --- Configuration (Inspired by Live Script) ---
RAW_SEQ_WINDOW = 100
PRICE_HISTORY_WINDOW = 100
TIME_HORIZON_SECONDS = 30.0 
FEE_PCT = 0.0004
WARMUP_MINUTES = 3.0

# Dynamic Barrier Configuration
K_VOL = 0.40
PROFIT_TAKE_MULT = 1.2
STOP_LOSS_MULT = 1.0
RETURNS_WINDOW_SIZE = 200
MIN_PROFIT_PCT = 0.001

def generate_feature_health_report(df_features: pd.DataFrame):
    # ... (function remains the same)
    print("\n" + "="*50)
    print(" " * 15 + "Feature Health Report")
    print("="*50)
    
    feature_cols = [c for c in df_features.columns if c not in ['time']]
    stats = df_features[feature_cols].describe().transpose()
    stats['abs_mean'] = stats['mean'].abs()
    
    print("\nFeature Statistics:")
    print(stats[['mean', 'std', 'min', 'max']].round(4).to_string())
    
    print("\nTop 5 Features by Absolute Mean Value (Potential Drivers):")
    print(stats.sort_values('abs_mean', ascending=False).head(5)[['mean', 'std']].round(4).to_string())
    print("="*50 + "\n")


def build_dataset(trade_path: str, book_path: str, output_dir: str):
    """
    Main function to run the data generation pipeline.
    """
    print("Step 1: Loading and merging raw data...")
    df = load_and_merge(trade_path, book_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Step 2: Simulating event stream (warming up for {WARMUP_MINUTES} minutes)...")
    
    # --- State Initialization ---
    # ... (state initialization remains the same)
    state = {
        "cvd_history": deque(), "price_history": deque(maxlen=PRICE_HISTORY_WINDOW),
        "welford_spread": WelfordStats(), "welford_rv": WelfordStats(),
        "welford_cvd_5s": WelfordStats(), "welford_cvd_1m": WelfordStats(), "welford_cvd_3m": WelfordStats(),
        "mid_price_history": deque(maxlen=50), "welford_rv_tactical": WelfordStats(),
    }
    raw_history = {
        "microprice": deque(maxlen=RAW_SEQ_WINDOW), "volume": deque(maxlen=RAW_SEQ_WINDOW),
        "spread": deque(maxlen=RAW_SEQ_WINDOW)
    }
    returns_window = deque(maxlen=RETURNS_WINDOW_SIZE)
    last_microprice = None
    cumulative_cvd = 0.0
    human_features_list, raw_sequences_list, labels_list = [], [], []

    # --- Main Simulation Loop ---
    trigger_indices = df[df["stream_type"] == "book"].index
    start_time_ms = df.iloc[0]["time"]
    warmup_duration_ms = WARMUP_MINUTES * 60 * 1000
    warmup_end_time_ms = start_time_ms + warmup_duration_ms
    
    ### NEW: State for trade-based sampling ###
    trades_since_last_sample = 0

    for i in tqdm(trigger_indices, desc="Generating Samples"):
        current_time_ms = df.iloc[i]["time"]
        
        # --- Process state (always runs to keep stats up-to-date) ---
        volume_since_last_book = 0.0
        start_index = state.get("last_processed_index", -1) + 1
        for j in range(start_index, i + 1):
            row = df.iloc[j]
            if row["stream_type"] == "trade" and not np.isnan(row["quote_qty"]):
                sign = 1 if not row["is_buyer_maker"] else -1
                cumulative_cvd += sign * row["quote_qty"]
                state["cvd_history"].append((row["time"] / 1000.0, cumulative_cvd))
                volume_since_last_book += row["quote_qty"]
                ### NEW: Increment trade counter ###
                trades_since_last_sample += 1
        state["last_processed_index"] = i
        
        current_book = df.iloc[i]
        state.update(current_book.to_dict())

        # Update raw history with the latest book state
        bid, ask = current_book["best_bid_price"], current_book["best_ask_price"]
        bid_q, ask_q = current_book["best_bid_qty"], current_book["best_ask_qty"]
        if not (np.isnan(bid) or np.isnan(ask) or np.isnan(bid_q) or np.isnan(ask_q)):
            microprice = (bid * ask_q + ask * bid_q) / (bid_q + ask_q + 1e-9)
            raw_history["microprice"].append(microprice)
            raw_history["spread"].append(ask - bid)
            raw_history["volume"].append(volume_since_last_book)
            
            mid_price = 0.5 * (bid + ask)
            state["mid_price_history"].append(mid_price)

            if last_microprice is not None and microprice > 0 and last_microprice > 0:
                ret = (microprice - last_microprice) / last_microprice
                returns_window.append(ret)
            last_microprice = microprice
        
        ### MODIFIED: Generate sample only if a trade has occurred ###
        conditions_met = (
            current_time_ms > warmup_end_time_ms and
            len(raw_history["microprice"]) == RAW_SEQ_WINDOW and
            len(returns_window) >= 50 and
            trades_since_last_sample > 0  # The crucial new condition
        )

        if conditions_met:
            # --- Reset the trade counter after deciding to sample ---
            trades_since_last_sample = 0

            # 1. Generate Human Features
            features = calculate_features(state)
            if not features: continue

            # 2. Generate Label with DYNAMIC barriers
            vol_now = np.std(returns_window)
            dynamic_barrier_width = K_VOL * vol_now
            final_barrier_width = max(MIN_PROFIT_PCT, dynamic_barrier_width)
            final_pt_pct = final_barrier_width * PROFIT_TAKE_MULT
            final_sl_pct = final_barrier_width * STOP_LOSS_MULT
            
            label, ret, t_hit = find_triple_barrier_label(
                df, i, final_pt_pct, final_sl_pct, TIME_HORIZON_SECONDS, FEE_PCT
            )

            # 3. Generate Raw Sequence for NN
            sequence = np.stack([
                np.array(raw_history["microprice"], dtype=np.float32),
                np.array(raw_history["volume"], dtype=np.float32),
                np.array(raw_history["spread"], dtype=np.float32)
            ], axis=1)

            # Store results
            human_features_list.append(features)
            raw_sequences_list.append(sequence)
            labels_list.append({
                "time": features["time"], 
                "label": label, 
                "ret": ret, 
                "time_to_hit": t_hit
            })

    # ... (Rest of the script for saving data and reporting remains the same)
    print("Step 3: Saving datasets to disk...")
    df_hf = pd.DataFrame(human_features_list)
    df_labels = pd.DataFrame(labels_list)
    
    if not df_hf.empty and not df_labels.empty:
        df_hf = df_hf.set_index('time')
        df_labels = df_labels.set_index('time')
        common_index = df_hf.index.intersection(df_labels.index)
        df_hf = df_hf.loc[common_index].reset_index()
        df_labels = df_labels.loc[common_index].reset_index()
        raw_sequences_list = np.array(raw_sequences_list, dtype=np.float32)[df_hf.index]

    save_parquet(df_hf, os.path.join(output_dir, "human_features.parquet"))
    save_parquet(df_labels, os.path.join(output_dir, "labels.parquet"))
    save_npy(raw_sequences_list, os.path.join(output_dir, "raw_sequences.npy"))

    if not df_hf.empty:
        generate_feature_health_report(df_hf)

    print(f"✅ Pipeline complete. Datasets saved in '{output_dir}'.")
    print(f"Total samples generated: {len(df_hf)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature and label datasets from raw tick data.")
    # ... (args remain the same)
    args = parser.parse_args()
    build_dataset(args.trade_csv, args.book_csv, args.out_dir)
