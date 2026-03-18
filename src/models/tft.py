from typing import List
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


def build_tft(max_encoder_length: int, max_prediction_length: int, quantiles: List[float], hidden_size: int = 64, attention_heads: int = 4, dropout: float = 0.1):
    return TemporalFusionTransformer.from_dataset(
        None,  # to be set via dataset.from_dataset in train script
        learning_rate=1e-3,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=dropout,
        loss=QuantileLoss(quantiles=quantiles),
        output_size=len(quantiles),
        log_interval=50,
        reduce_on_plateau_patience=3,
    )
