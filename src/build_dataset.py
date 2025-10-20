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

# Configuration
RAW_SEQ_WINDOW = 100
PRICE_HISTORY_WINDOW = 100
PROFIT_TAKE_PCT = 0.0015
STOP_LOSS_PCT = 0.0015
LOOKAHEAD_ROWS = 500
FEE_PCT = 0.0004 # 0.04% taker fee

def build_dataset(trade_path: str, book_path: str, output_dir: str):
    """
    Main function to run the data generation pipeline.
    """
    print("Step 1: Loading and merging raw data...")
    df = load_and_merge(trade_path, book_path)
    os.makedirs(output_dir, exist_ok=True)

    print("Step 2: Simulating event stream to generate features and labels...")
    
    # --- State Initialization ---
    state = {
        "cvd_history": deque(),
        "price_history": deque(maxlen=PRICE_HISTORY_WINDOW),
        "welford_spread": WelfordStats(),
        "welford_rv": WelfordStats(),
        "welford_cvd_5s": WelfordStats(),
        "welford_cvd_1m": WelfordStats(),
        "welford_cvd_3m": WelfordStats(),
    }
    # Raw history for NN sequences
    raw_history = {
        "microprice": deque(maxlen=RAW_SEQ_WINDOW),
        "volume": deque(maxlen=RAW_SEQ_WINDOW),
        "spread": deque(maxlen=RAW_SEQ_WINDOW)
    }
    
    # Cumulative CVD state
    cumulative_cvd = 0.0

    # Lists to store final data
    human_features_list = []
    raw_sequences_list = []
    labels_list = []

    # --- Main Simulation Loop ---
    trigger_indices = df[df["stream_type"] == "book"].index
    for i in tqdm(trigger_indices, desc="Generating Samples"):
        # Fast-forward state to current index `i`
        start_index = state.get("last_processed_index", -1) + 1
        for j in range(start_index, i + 1):
            row = df.iloc[j]
            if row["stream_type"] == "trade" and not np.isnan(row["quote_qty"]):
                # CORRECTED CVD: taker is buyer -> positive, taker is seller -> negative
                sign = 1 if not row["is_buyer_maker"] else -1
                cumulative_cvd += sign * row["quote_qty"]
                state["cvd_history"].append((row["time"] / 1000.0, cumulative_cvd))
                raw_history["volume"].append(row["quote_qty"])

        state["last_processed_index"] = i
        
        # Current book state for feature calculation
        current_book = df.iloc[i]
        state.update(current_book.to_dict())

        # Update raw history with book data
        bid, ask = current_book["best_bid_price"], current_book["best_ask_price"]
        bid_q, ask_q = current_book["best_bid_qty"], current_book["best_ask_qty"]
        if not (np.isnan(bid) or np.isnan(ask) or np.isnan(bid_q) or np.isnan(ask_q)):
            microprice = (bid * ask_q + ask * bid_q) / (bid_q + ask_q + 1e-9)
            raw_history["microprice"].append(microprice)
            raw_history["spread"].append(ask - bid)
        
        # --- Generate Sample Packet ---
        if len(raw_history["microprice"]) == RAW_SEQ_WINDOW:
            # 1. Generate Human Features
            features = calculate_features(state)
            if not features: continue

            # 2. Generate Label
            label, ret, t_hit = find_triple_barrier_label(
                df, i, PROFIT_TAKE_PCT, STOP_LOSS_PCT, LOOKAHEAD_ROWS, FEE_PCT
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

    print("Step 3: Saving datasets to disk...")
    df_hf = pd.DataFrame(human_features_list)
    df_labels = pd.DataFrame(labels_list)
    
    # Align all datasets by the timestamp from features
    df_labels = df_labels.set_index('time').loc[df_hf['time']].reset_index()
    
    save_parquet(df_hf, os.path.join(output_dir, "human_features.parquet"))
    save_parquet(df_labels, os.path.join(output_dir, "labels.parquet"))
    save_npy(np.array(raw_sequences_list, dtype=np.float32), os.path.join(output_dir, "raw_sequences.npy"))

    print(f"✅ Pipeline complete. Datasets saved in '{output_dir}'.")
    print(f"Total samples generated: {len(df_hf)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature and label datasets from raw tick data.")
    parser.add_argument("--trade-csv", required=True, help="Path to the trade data CSV.")
    parser.add_argument("--book-csv", required=True, help="Path to the book ticker data CSV.")
    parser.add_argument("--out-dir", required=True, help="Directory to save the output parquet and npy files.")
    args = parser.parse_args()
    build_dataset(args.trade_csv, args.book_csv, args.out_dir)
      
