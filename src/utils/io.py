
"""
I/O helpers: save/load parquet and numpy arrays and metadata.
"""

import pandas as pd
import numpy as np
import json
import os

def save_parquet(df, path, compression="snappy"):
    df = pd.DataFrame(df)
    df.to_parquet(path, compression=compression)
    return path

def save_npy(arr, path):
    np.save(path, arr)
    return path

def load_parquet(path):
    return pd.read_parquet(path)

def load_npy(path):
    return np.load(path, allow_pickle=False)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
      
