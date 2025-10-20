import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import argparse
from src.models.transformer_arch import TransformerFeatureExtractor
from sklearn.metrics import classification_report

class Predictor(nn.Module):
    """Wrapper model for training: combines feature extractor and a classification head."""
    def __init__(self, feature_extractor, output_dim, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = nn.Linear(output_dim, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classification_head(features)
        return logits

def train(seq_npy, labels_parquet, output_path="transformer_feature_extractor.pth", 
          epochs=20, bs=128, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    print(f"Using device: {device}")
    X = np.load(seq_npy).astype(np.float32)
    # Labels need to be shifted for CrossEntropyLoss: {-1, 0, 1} -> {0, 1, 2}
    y = pd.read_parquet(labels_parquet)["label"].values.astype(np.int64) + 1
    
    # --- UPDATED: Full Train/Validation/Test Split ---
    n = len(X)
    i1 = int(n * 0.7)
    i2 = int(n * 0.85)
    X_train, X_val, X_test = X[:i1], X[i1:i2], X[i2:]
    y_train, y_val, y_test = y[:i1], y[i1:i2], y[i2:]

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # --- Model Config (should be stored in a config file) ---
    model_config = {
        "input_dim": X.shape[2], "d_model": 64, "n_heads": 4,
        "dim_feedforward": 256, "num_layers": 2, "output_dim": 32, "max_seq_len": 100
    }
    feature_extractor = TransformerFeatureExtractor(**model_config)
    model = Predictor(feature_extractor, model_config["output_dim"], num_classes=3).to(device)

    # --- Modern Training Stack ---
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5

    print("Starting training...")
    for e in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                total_val_loss += loss.item() * xb.shape[0]
        
        avg_val_loss = total_val_loss / len(val_dataset)
        print(f"Epoch {e+1}/{epochs}: Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.feature_extractor.state_dict(), output_path)
            print(f"Model saved to {output_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break
    
    print("\n--- Final Evaluation on Test Set ---")
    # Load the best model for final evaluation
    best_feature_extractor = TransformerFeatureExtractor(**model_config)
    best_feature_extractor.load_state_dict(torch.load(output_path, map_location=device))
    final_model = Predictor(best_feature_extractor, model_config["output_dim"], num_classes=3).to(device)
    final_model.eval()
    
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = final_model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())
            
    print(classification_report(all_true, all_preds, target_names=["Short Win", "Neutral/Loss", "Long Win"]))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the Transformer feature extractor.")
    p.add_argument("--seq", required=True, help="Path to raw_sequences.npy")
    p.add_argument("--labels", required=True, help="Path to labels.parquet")
    p.add_argument("--out", default="transformer_feature_extractor.pth", help="Path to save the trained model weights.")
    args = p.parse_args()
    train(args.seq, args.labels, args.out)

