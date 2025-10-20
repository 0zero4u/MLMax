# src/ingest/loader.py
"""
Fast loader + merge for trade + book CSVs. Exposes `load_and_merge`.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from src.ingest.validate import validate_trade_csv, validate_book_csv

def load_and_merge(trade_path: str, book_path: str, symbol:str="BTCUSDT") -> pd.DataFrame:
    trades = validate_trade_csv(trade_path)
    books = validate_book_csv(book_path)

    trades = trades.copy()
    books = books.copy()

    trades["stream_type"] = "trade"
    books["stream_type"] = "book"

    # Derived
    trades["quote_qty"] = trades["quote_qty"].fillna(trades["price"] * trades["qty"])
    books["spread"] = books["best_ask_price"] - books["best_bid_price"]
    books["mid_price"] = (books["best_ask_price"] + books["best_bid_price"]) / 2

    # Ensure consistent columns across both frames
    for col in ["price","qty","quote_qty","is_buyer_maker"]:
        if col not in books.columns:
            books[col] = np.nan
    for col in ["best_bid_price","best_bid_qty","best_ask_price","best_ask_qty","spread","mid_price"]:
        if col not in trades.columns:
            trades[col] = np.nan

    df = pd.concat([trades, books], ignore_index=True, sort=False)
    df = df.sort_values("time").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")
    df["symbol"] = symbol
    return df
