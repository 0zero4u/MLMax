# src/models/train_lgbm.py
"""
Simple LightGBM training script that loads human_features.parquet + labels.
Time-based split is used.
"""

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import json
from src.utils.io import load_parquet, save_json

def train(hf_path, labels_path, output_model_path="lgbm_model.txt"):
    X = pd.read_parquet(hf_path).sort_values("time").reset_index(drop=True)
    y = pd.read_parquet(labels_path).sort_values("time").reset_index(drop=True)["label"]
    # drop time column
    times = X["time"].values
    X = X.drop(columns=["time"], errors="ignore")
    # train/val/test split (70/15/15 by time)
    n = len(X)
    i1 = int(n*0.7); i2 = int(n*0.85)
    X_train, X_val, X_test = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
    y_train, y_val, y_test = y.iloc[:i1], y.iloc[i1:i2], y.iloc[i2:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "num_leaves": 64,
        "learning_rate": 0.05
    }

    bst = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=500, early_stopping_rounds=50)
    bst.save_model(output_model_path)

    # eval
    preds = bst.predict(X_test)
    preds_labels = preds.argmax(axis=1)
    print(classification_report(y_test, preds_labels))
    # save metrics
    save_json({
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test)
    }, "train_metadata.json")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", default="lgbm_model.txt")
    args = p.parse_args()
    train(args.hf, args.labels, args.out)
  
