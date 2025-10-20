import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error
import joblib
import argparse

def train(hf_path, nf_path, labels_path, out_clf="lgbm_classifier.joblib", out_reg="lgbm_regressor.joblib"):
    # 1. Merge Datasets
    print("Loading and merging datasets...")
    df_hf = pd.read_parquet(hf_path)
    df_nf = pd.read_parquet(nf_path)
    df_labels = pd.read_parquet(labels_path)

    X = pd.concat([df_hf.drop(columns=['time']), df_nf], axis=1)
    y_df = df_labels.set_index("time").sort_index()
    X.index = df_hf['time']
    X = X.sort_index()
    
    # Ensure perfect alignment
    common_index = X.index.intersection(y_df.index)
    X = X.loc[common_index]
    y_df = y_df.loc[common_index]
    
    y_clf = y_df["label"]
    y_reg = y_df["ret"]
    print(f"Final aligned dataset shape: {X.shape}")

    # 2. Train/Test Split (70/15/15 by time)
    n = len(X)
    i1 = int(n * 0.7); i2 = int(n * 0.85)
    X_train, X_val, X_test = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
    y_clf_train, y_clf_val, y_clf_test = y_clf.iloc[:i1], y_clf.iloc[i1:i2], y_clf.iloc[i2:]
    y_reg_train, y_reg_val, y_reg_test = y_reg.iloc[:i1], y_reg.iloc[i1:i2], y_reg.iloc[i2:]

    # 3. Train Classifier
    print("\n--- Training LGBM Classifier ---")
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_clf_train, eval_set=[(X_val, y_clf_val)], 
            callbacks=[lgb.early_stopping(100, verbose=True)])
    joblib.dump(clf, out_clf)
    print(f"Classifier saved to {out_clf}")
    
    print("\nClassifier Evaluation on Test Set:")
    preds_labels = clf.predict(X_test)
    # Adjust labels for report: {-1, 0, 1} -> {0, 1, 2} and then map to names
    print(classification_report(y_clf_test, preds_labels, target_names=["Short Win (-1)", "Neutral/Loss (0)", "Long Win (1)"]))

    # 4. Train Regressor
    print("\n--- Training LGBM Regressor ---")
    # --- UPDATED: Train ONLY on Win conditions (label is not 0) ---
    win_indices_train = y_clf_train[y_clf_train != 0].index
    win_indices_val = y_clf_val[y_clf_val != 0].index
    win_indices_test = y_clf_test[y_clf_test != 0].index
    
    X_reg_train, y_reg_train = X.loc[win_indices_train], y_reg.loc[win_indices_train]
    X_reg_val, y_reg_val = X.loc[win_indices_val], y_reg.loc[win_indices_val]
    X_reg_test, y_reg_test = X.loc[win_indices_test], y_reg.loc[win_indices_test]

    if len(X_reg_train) > 0:
        reg = lgb.LGBMRegressor(objective="regression_l1", n_estimators=1000, random_state=42, n_jobs=-1)
        reg.fit(X_reg_train, y_reg_train, eval_set=[(X_reg_val, y_reg_val)], 
                callbacks=[lgb.early_stopping(100, verbose=True)])
        joblib.dump(reg, out_reg)
        print(f"Regressor saved to {out_reg}")

        print("\nRegressor Evaluation on Test Set (Win Conditions Only):")
        preds_reg = reg.predict(X_reg_test)
        mae = mean_absolute_error(y_reg_test, preds_reg)
        print(f"Mean Absolute Error: {mae:.8f}")
    else:
        print("Not enough non-zero labels to train a regressor.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train final LightGBM models.")
    p.add_argument("--hf", required=True, help="Path to human_features.parquet")
    p.add_argument("--nf", required=True, help="Path to neural_features.parquet")
    p.add_argument("--labels", required=True, help="Path to labels.parquet")
    p.add_argument("--out-clf", default="lgbm_classifier.joblib")
    p.add_argument("--out-reg", default="lgbm_regressor.joblib")
    args = p.parse_args()
    train(args.hf, args.nf, args.labels, args.out_clf, args.out_reg)
