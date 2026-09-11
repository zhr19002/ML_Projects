import torch
from config import *
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MultiHorizonMetric

class RainOccurrenceLoss(MultiHorizonMetric):
    def loss(self, y_pred, target):
        logits = self.to_prediction(y_pred)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')

class IntensityWeightedHuberLoss(MultiHorizonMetric):
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def loss(self, y_pred, target):
        y_pred = self.to_prediction(y_pred)
        rain_mask = target > RAIN_THRESHOLD
        error = (y_pred - target).abs()
        quadratic = torch.minimum(error, torch.tensor(self.delta, device=error.device))
        huber = 0.5 * quadratic**2 + self.delta * (error - quadratic)
        weight = 1 + torch.log1p(target)
        return huber * weight * rain_mask.float()

def build_rain_model(dataset):
    model = TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size = HIDDEN_SIZE,
        lstm_layers = LSTM_LAYERS,
        dropout = DROPOUT,
        attention_head_size = ATTENTION_HEAD_SIZE,
        hidden_continuous_size = HIDDEN_CONTINUOUS_SIZE,
        learning_rate = LEARNING_RATE0,
        loss = RainOccurrenceLoss(),
        optimizer = 'adam',
        weight_decay = 1e-5,
        reduce_on_plateau_patience = 5,
        log_interval = -1,
    )
    return model

def build_amount_model(dataset):
    model = TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size = HIDDEN_SIZE,
        lstm_layers = LSTM_LAYERS,
        dropout = DROPOUT,
        attention_head_size = ATTENTION_HEAD_SIZE,
        hidden_continuous_size = HIDDEN_CONTINUOUS_SIZE,
        learning_rate = LEARNING_RATE0,
        loss = IntensityWeightedHuberLoss(),
        optimizer = 'adam',
        weight_decay = 1e-5,
        reduce_on_plateau_patience = 5,
        log_interval = -1,
    )
    return model

def load_model(ckpt):
    return TemporalFusionTransformer.load_from_checkpoint(ckpt)