import os
from config import *
from utils import *
from preprocess import load_all_stations, split_source_target, split_by_year
from data_module import build_source_set, build_dataloaders, build_target_set
from model import load_model
from train import build_trainer

def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'output_layer' in name or 'pre_output_gate_norm' in name:
            param.requires_grad = True

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def finetune_stage(stage_name, source_ckpt, train_loader, val_loader, filename1, filename2):
    # Stage 1: freeze backbone
    print(f'** {stage_name}: freeze backbone **')
    source_tft = load_model(os.path.join(CKPT_DIR, source_ckpt))
    freeze_backbone(source_tft)
    source_tft.hparams.learning_rate = LEARNING_RATE1
    count_parameters(source_tft)
    trainer1 = build_trainer(filename1, MAX_EPOCHS1, PATIENCE1)
    trainer1.fit(source_tft, train_loader, val_loader)

    # Stage 2: unfreeze all
    print(f'\n** {stage_name}: unfreeze all **')
    target_tft1 = load_model(os.path.join(CKPT_DIR, f'{filename1}.ckpt'))
    unfreeze_all(target_tft1)
    target_tft1.hparams.learning_rate = LEARNING_RATE2
    count_parameters(target_tft1)
    trainer2 = build_trainer(filename2, MAX_EPOCHS2, PATIENCE2)
    trainer2.fit(target_tft1, train_loader, val_loader)

def main():
    # Load data
    print_header('Load data')
    all_df = load_all_stations(DATA_DIR)
    source_df, target_df = split_source_target(all_df)
    source_train, _, _ = split_by_year(source_df)
    target_train, target_val, target_test = split_by_year(target_df)
    print(f'Train samples: {len(target_train):,}')
    print(f'Val samples  : {len(target_val):,}')
    print(f'Test samples : {len(target_test):,}\n')

    # Stage 1: rain occurrence
    print_header('Build finetuned rain occurrence TFT')
    source_set_rain = build_source_set(source_train, stage=1)
    target_set_rain = build_target_set(source_set_rain, target_train)
    train_loader_rain, val_loader_rain, _ = build_dataloaders(target_set_rain, target_val, target_test, BATCH_SIZE1)
    
    # Fine-tune rain occurrence TFT
    finetune_stage(
        stage_name = 'Rain occurrence TFT',
        source_ckpt = f'{FILENAME0}_rain.ckpt',
        train_loader = train_loader_rain,
        val_loader = val_loader_rain,
        filename1 = f'{FILENAME1}_rain_{START_YEAR}',
        filename2 = f'{FILENAME2}_rain_{START_YEAR}',
    )

    # Stage 2: rain amount
    print_header('Build finetuned rain amount TFT')
    source_set_amount = build_source_set(source_train, stage=2)
    target_set_amount = build_target_set(source_set_amount, target_train)
    train_loader_amount, val_loader_amount, _ = build_dataloaders(target_set_amount, target_val, target_test, BATCH_SIZE1)

    # Fine-tune rain amount TFT
    finetune_stage(
        stage_name = 'Rain amount TFT',
        source_ckpt = f'{FILENAME0}_amount.ckpt',
        train_loader = train_loader_amount,
        val_loader = val_loader_amount,
        filename1 = f'{FILENAME1}_amount_{START_YEAR}',
        filename2 = f'{FILENAME2}_amount_{START_YEAR}',
    )

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    seed_everything(42)
    main()