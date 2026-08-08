import os
import random
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

def print_header(title):
    print('=' * 70)
    print(title)
    print('=' * 70)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters    : {total:,}')
    print(f'Trainable parameters: {trainable:,}\n')

def save_experiment(csv_path, name, metrics):
    df = pd.DataFrame([{'Model': name, **metrics}])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)