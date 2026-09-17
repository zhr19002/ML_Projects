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
    _, target_df = split_source_target(all_df)
    target_train, target_val, target_test = split_by_year(target_df)
    print(f'Train samples: {len(target_train):,}')
    print(f'Val samples  : {len(target_val):,}')
    print(f'Test samples : {len(target_test):,}\n')

    # Stage 1: rain occurrence
    print_header('Build scratch rain occurrence TFT')
    target_set_rain = build_source_set(target_train, stage=1)
    train_loader_rain, val_loader_rain, _ = build_dataloaders(target_set_rain, target_val, target_test, BATCH_SIZE1)
    rain_tft = build_rain_model(target_set_rain)
    count_parameters(rain_tft)
    trainer_rain = build_trainer(f'{FILENAME3}_rain_{START_YEAR}', MAX_EPOCHS3, PATIENCE3)
    trainer_rain.fit(rain_tft, train_loader_rain, val_loader_rain)

    # Stage 2: Rain amount
    print_header('Build scratch rain amount TFT')
    target_set_amount = build_source_set(target_train, stage=2)
    train_loader_amount, val_loader_amount, _ = build_dataloaders(target_set_amount, target_val, target_test, BATCH_SIZE1)
    amount_tft = build_amount_model(target_set_amount)
    count_parameters(amount_tft)
    trainer_amount = build_trainer(f'{FILENAME3}_amount_{START_YEAR}', MAX_EPOCHS3, PATIENCE3)
    trainer_amount.fit(amount_tft, train_loader_amount, val_loader_amount)

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    seed_everything(42)
    main()