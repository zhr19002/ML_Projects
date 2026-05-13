import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataConfig:
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    cols: list

@dataclass
class ModelConfig:
    context: int
    horizon: int
    input_dim: int

@dataclass
class TrainConfig:
    train_preds: np.ndarray
    val_preds: np.ndarray