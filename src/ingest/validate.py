
import pandas as pd
import numpy as np

# These are the columns the rest of the program LOGICALLY needs.
# Our job in this file is to load the CSVs and rename/select their columns to match these.
REQUIRED_TRADE_COLUMNS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]
REQUIRED_BOOK_COLUMNS = ["time", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]

def _check_columns(df: pd.DataFrame, required: list):
    """Checks if a dataframe contains all required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after processing: {missing}")

def validate_trade_csv(path: str, max_price_err: float = 1e-6) -> pd.DataFrame:
    """
    Validates the trade data CSV. Assumes the file contains a header row.
    """
    # This file has a header, so we just read it normally.
    # pandas will automatically use the first row as column names.
    df = pd.read_csv(path)
    
    # Check if the expected columns exist in the loaded dataframe.
    _check_columns(df, REQUIRED_TRADE_COLUMNS)
    
    # --- Data Type Conversion and Sanity Checks ---
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["quote_qty"] = df["quote_qty"].astype(float)
    df["time"] = df["time"].astype("int64")
    # The boolean column can be strings like 'false'/'true'
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).map({
        "True": True, "true": True, 
        "False": False, "false": False, 
        "0": False, "1": True
    })
    
    # Sanity check: quote_qty should be approximately price * qty
    calc = df["price"] * df["qty"]
    diff = (calc - df["quote_qty"]).abs()
    bad = diff > max_price_err * (1 + df["quote_qty"].abs())
    if bad.any():
        # Don't crash — warn and keep rows, user can inspect
        print(f"[validate_trade_csv] WARNING: {bad.sum()} rows have quote_qty mismatch (first 5 indices): {list(df[bad].index[:5])}")
    
    return df

def validate_book_csv(path: str) -> pd.DataFrame:
    """
    Validates the book data CSV.
    This function is specifically adapted for the 7-column `@depth` stream format.
    It will load the data, rename the timestamp column, and select the required columns.
    """
    # 1. Define the ACTUAL column names from your specific file format.
    actual_book_columns = [
        "update_id", "best_bid_price", "best_bid_qty", 
        "best_ask_price", "best_ask_qty", "transaction_time", "event_time"
    ]
    
    # 2. Load the CSV. We use `skiprows=1` to ignore the header line in the file
    #    and `names=...` to programmatically assign the correct column names.
    df = pd.read_csv(path, header=None, names=actual_book_columns, skiprows=1)

    # 3. RENAME the 'transaction_time' column to 'time' so the rest of the script can find it.
    df = df.rename(columns={"transaction_time": "time"})

    # 4. SELECT only the columns the script needs, in the correct, expected order.
    #    This adapts the 7-column input to the 5-column format the program requires.
    df = df[REQUIRED_BOOK_COLUMNS]

    # This check will now pass because we have created the required columns.
    _check_columns(df, REQUIRED_BOOK_COLUMNS)
    
    # --- Data Type Conversion ---
    df["best_bid_price"] = df["best_bid_price"].astype(float)
    df["best_ask_price"] = df["best_ask_price"].astype(float)
    df["best_bid_qty"] = df["best_bid_qty"].astype(float)
    df["best_ask_qty"] = df["best_ask_qty"].astype(float)
    df["time"] = df["time"].astype("int64")
    
    return df
