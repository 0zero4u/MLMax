# src/models/train_transformer.py
"""
Minimal PyTorch training loop for sequences (skeleton).
Saves a trained encoder state dict. Intended as a starting point.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=3, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, emb_dim),
            nn.ReLU()
        )
    def forward(self, x):
        # x: (B, T, K) -> (B, K, T)
        x = x.permute(0,2,1)
        return self.net(x)

def train(seq_npy, labels_parquet, epochs=5, bs=64, lr=1e-3, device='cpu'):
    X = np.load(seq_npy).astype(np.float32)
    y = pd.read_parquet(labels_parquet)["label"].values.astype(np.int64)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    model = SimpleEncoder(input_dim=X.shape[2], emb_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for e in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
        print(f"Epoch {e}: loss={total_loss/len(dataset)}")
    torch.save(model.state_dict(), "encoder.pth")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seq", required=True)
    p.add_argument("--labels", required=True)
    args = p.parse_args()
    train(args.seq, args.labels)

