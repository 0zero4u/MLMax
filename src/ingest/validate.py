
import pandas as pd
import numpy as np

REQUIRED_TRADE_COLUMNS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]
REQUIRED_BOOK_COLUMNS = ["time", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]

def _check_columns(df: pd.DataFrame, required: list):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def validate_trade_csv(path: str, max_price_err: float = 1e-6) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_columns(df, REQUIRED_TRADE_COLUMNS)
    # dtypes
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["quote_qty"] = df["quote_qty"].astype(float)
    df["time"] = df["time"].astype("int64")
    # boolean column can be strings like 'false'/'true'
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).map({"True":True,"true":True,"False":False,"false":False,"0":False,"1":True})
    # sanity: quote ~= price * qty
    calc = df["price"] * df["qty"]
    diff = (calc - df["quote_qty"]).abs()
    bad = diff > max_price_err * (1 + df["quote_qty"].abs())
    if bad.any():
        # don't crash — warn and keep rows, user can inspect
        print(f"[validate_trade_csv] WARNING: {bad.sum()} rows have quote_qty mismatch (first 5 indices): {list(df[bad].index[:5])}")
    return df

def validate_book_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_columns(df, REQUIRED_BOOK_COLUMNS)
    df["best_bid_price"] = df["best_bid_price"].astype(float)
    df["best_ask_price"] = df["best_ask_price"].astype(float)
    df["best_bid_qty"] = df["best_bid_qty"].astype(float)
    df["best_ask_qty"] = df["best_ask_qty"].astype(float)
    df["time"] = df["time"].astype("int64")
    return df
  
