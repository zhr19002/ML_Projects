import os
from config import *
from utils import *
from preprocess import load_all_stations, split_source_target, split_by_year
from data_module import build_source_set, build_dataloaders
from model import build_rain_model, build_amount_model
from train import build_trainer

def main():
    # Load data
    print_header('Load data')
    all_df = load_all_stations(DATA_DIR)
    source_df, _ = split_source_target(all_df)
    source_train, source_val, source_test = split_by_year(source_df)
    print(f'Train samples: {len(source_train):,}')
    print(f'Val samples  : {len(source_val):,}')
    print(f'Test samples : {len(source_test):,}\n')

    # Stage 1: rain occurrence
    print_header('Build pretrained rain occurrence TFT')
    source_set_rain = build_source_set(source_train, stage=1)
    train_loader_rain, val_loader_rain, _ = build_dataloaders(source_set_rain, source_val, source_test, BATCH_SIZE0)
    rain_tft = build_rain_model(source_set_rain)
    count_parameters(rain_tft)
    trainer_rain = build_trainer(f'{FILENAME0}_rain', MAX_EPOCHS0, PATIENCE0)
    trainer_rain.fit(rain_tft, train_loader_rain, val_loader_rain)

    # Stage 2: rain amount
    print_header('Build pretrained rain amount TFT')
    source_set_amount = build_source_set(source_train, stage=2)
    train_loader_amount, val_loader_amount, _ = build_dataloaders(source_set_amount, source_val, source_test, BATCH_SIZE0)
    amount_tft = build_amount_model(source_set_amount)
    count_parameters(amount_tft)
    trainer_amount = build_trainer(f'{FILENAME0}_amount', MAX_EPOCHS0, PATIENCE0)
    trainer_amount.fit(amount_tft, train_loader_amount, val_loader_amount)

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    seed_everything(42)
    main()