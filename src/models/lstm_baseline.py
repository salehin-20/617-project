"""
Lightweight LSTM baseline for 24-hour ahead load forecasting.
Features: y, temp, holiday, extreme_flag. Target: next 24h of y with quantiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
import pytorch_lightning as pl


def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    losses = []
    for i, q in enumerate(quantiles):
        e = target - preds[..., i]
        losses.append(torch.max((q - 1) * e, q * e))
    return torch.mean(torch.stack(losses, dim=-1))


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    lookback: int = 24 * 7
    horizon: int = 24
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)


class LSTMBaseline(pl.LightningModule):
    def __init__(self, cfg: LSTMConfig, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.lr = lr

        self.encoder = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.horizon * len(cfg.quantiles)),
        )

    def forward(self, x):
        enc_out, _ = self.encoder(x)  # x: (batch, lookback, input)
        h_last = enc_out[:, -1, :]
        out = self.proj(h_last)
        out = out.view(-1, self.cfg.horizon, len(self.cfg.quantiles))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = quantile_loss(preds, y, list(self.cfg.quantiles))
        self.log("train_q_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = quantile_loss(preds, y, list(self.cfg.quantiles))
        mae = torch.mean(torch.abs(preds[..., 1] - y))
        self.log("val_q_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
