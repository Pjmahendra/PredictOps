"""Deep learning experiments: LSTM and 1D-CNN on raw sensor sequences (PyTorch).

Input  : sequences of SEQ_LEN consecutive 1-minute rows x 13 channels (minute means of
         6 analog sensors + share-active of 7 digital signals). No hand-made features —
         the network must learn temporal patterns itself.
Target : label of the LAST minute of the sequence (y_detect). The sequence contains
         only that minute and the past, so there is no look-ahead.
Rules  : sequences never cross a recording gap (segment_id); every minute must be usable;
         train/validation/test sequences come from their own splits, which are separated by
         a 12 h embargo (> SEQ_LEN), so no sequence overlaps two partitions.
Imbalance: BCEWithLogitsLoss(pos_weight = negatives/positives) — sequence-preserving;
           SMOTE on sequences would break temporal structure (docs/04 §5.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

CHANNELS = ["TP2", "TP3", "H1", "DV_pressure", "Oil_temperature", "Motor_current",
            "COMP", "DV_eletric", "Towers", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
SEQ_LEN = 60  # minutes


def make_sequences(m: pd.DataFrame, y_col: str, stride: int, mean: np.ndarray, std: np.ndarray,
                   seq_len: int = SEQ_LEN):
    """Return X (n, seq_len, channels) float32, y (n,), end timestamps (n,)."""
    Xs, ys, ts = [], [], []
    for _, g in m.groupby("segment_id", sort=False):
        if len(g) < seq_len:
            continue
        v = ((g[CHANNELS].to_numpy(np.float32) - mean) / std)
        v = np.nan_to_num(v)
        ok = g["usable"].to_numpy() == 1
        y = g[y_col].to_numpy(np.float32)
        t = g["timestamp"].to_numpy()
        # windows ending at i (inclusive); require all minutes usable
        bad = np.convolve(~ok, np.ones(seq_len, dtype=int), "valid") > 0
        ends = np.arange(seq_len - 1, len(g))[~bad][::stride]
        if len(ends) == 0:
            continue
        idx = ends[:, None] - np.arange(seq_len)[::-1][None, :]
        Xs.append(v[idx]); ys.append(y[ends]); ts.append(t[ends])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ts)


class LSTMNet(nn.Module):
    """2-layer LSTM -> last hidden state -> linear. Learns long-range order dependencies."""
    def __init__(self, n_ch=len(CHANNELS), hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class CNN1D(nn.Module):
    """Two Conv1d blocks + global average pooling. Learns local temporal shapes
    (e.g. a missing load/unload oscillation) with few parameters."""
    def __init__(self, n_ch=len(CHANNELS), filters=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, filters, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(filters),
            nn.MaxPool1d(2),
            nn.Conv1d(filters, filters, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(filters),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(filters, 1))

    def forward(self, x):                        # x: (batch, time, channels)
        return self.net(x.transpose(1, 2)).squeeze(-1)


def predict(model, X, device, batch=2048):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).to(device))).cpu().numpy())
    return np.concatenate(out)


def train_model(model, Xtr, ytr, Xva, yva, device, epochs=30, patience=5, lr=1e-3, batch=256, seed=42):
    """Adam + class-weighted BCE + early stopping on validation loss. Returns history."""
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xv, yv = torch.from_numpy(Xva).to(device), torch.from_numpy(yva).to(device)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(Xtr)); tl = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            xb, yb = torch.from_numpy(Xtr[b]).to(device), torch.from_numpy(ytr[b]).to(device)
            opt.zero_grad(); l = loss_fn(model(xb), yb); l.backward(); opt.step(); tl += l.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(model(Xv[i:i + 2048]), yv[i:i + 2048]).item() for i in range(0, len(Xv), 2048)]))
        hist.append({"epoch": ep + 1, "train_loss": tl / len(perm), "val_loss": vl})
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return hist
