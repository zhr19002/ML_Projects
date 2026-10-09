import os
import joblib
import torch
import numpy as np
from config import *
from utils import *
from preprocess import load_all_stations, split_source_target, split_by_year
from data_module import build_source_set, build_dataloaders, build_target_set
from model import load_model
from calibrate import TwoStageRainCalibrator
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_two_stage_prediction(rain_model, amount_model, rain_loader, amount_loader):
    rain_model.eval()
    amount_model.eval()
    with torch.no_grad():
        # Stage 1: rain occurrence
        rain_logits = rain_model.predict(rain_loader, mode='prediction')
        rain_logits = rain_logits.detach().cpu().numpy()
        rain_prob = torch.sigmoid(torch.from_numpy(rain_logits)).numpy()
        # Stage 2: rain amount
        pred = amount_model.predict(amount_loader, mode='prediction', return_y=True)
        amount_pred = pred.output.detach().cpu().numpy()
        amount_pred = np.maximum(amount_pred, 0.0)
        y_true = pred.y[0].detach().cpu().numpy()
    return rain_prob, amount_pred, y_true

def categorical_metrics(y_true, y_flag, idx, threshold, tolerance):
    obs = (y_true[:, idx] > threshold)
    pred = y_flag[:, idx].astype(bool)

    # Tolerance window of this lead time
    start = max(0, idx - tolerance)
    end = min(y_true.shape[1], idx + tolerance + 1)
    obs_window = (y_true[:, start:end] > threshold)
    pred_window = y_flag[:, start:end].astype(bool)
    obs_match = np.any(obs_window, axis=1)
    pred_match = np.any(pred_window, axis=1)
    
    tp, fp, fn = np.sum(obs_match & pred), np.sum(~obs_match & pred), np.sum(obs & ~pred_match)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    return {'CSI': csi, 'POD': pod, 'FAR': far}

def compute_metrics(y_true, y_pred, rain_flag, alarm_flag, idx, threshold, tolerance):
    # Continuous metrics
    true_cum = np.sum(y_true[:, :idx+1], axis=1)
    pred_cum = np.sum(y_pred[:, :idx+1], axis=1)
    mae_cum = mean_absolute_error(true_cum, pred_cum)
    rmse_cum = np.sqrt(mean_squared_error(true_cum, pred_cum))
    bias_cum = np.mean(pred_cum - true_cum)
    
    # Categorical metrics
    occur = categorical_metrics(y_true, rain_flag, idx, RAIN_THRESHOLD, tolerance)
    alarm = categorical_metrics(y_true, alarm_flag, idx, threshold, tolerance)

    return {'MAE_cum': mae_cum, 'RMSE_cum': rmse_cum, 'Bias_cum': bias_cum,
            'CSI_occur': occur['CSI'], 'POD_occur': occur['POD'], 'FAR_occur': occur['FAR'],
            'CSI_alarm': alarm['CSI'], 'POD_alarm': alarm['POD'], 'FAR_alarm': alarm['FAR']}

def main():
    # Load data
    all_df = load_all_stations(DATA_DIR)
    source_df, target_df = split_source_target(all_df)
    source_train, _, _ = split_by_year(source_df)
    target_train, target_val, target_test = split_by_year(target_df)
    threshold = target_df.loc[target_df['prcp'] > RAIN_THRESHOLD, 'prcp'].quantile(QUANTILE)

    # Build stage 1: rain occurrence dataset
    source_set_rain = build_source_set(source_train, stage=1)
    target_set_rain = build_target_set(source_set_rain, target_train)
    _, val_loader_rain, test_loader_rain = build_dataloaders(target_set_rain, target_val, target_test, BATCH_SIZE1)

    # Build stage 2: rain amount dataset
    source_set_amount = build_source_set(source_train, stage=2)
    target_set_amount = build_target_set(source_set_amount, target_train)
    _, val_loader_amount, test_loader_amount = build_dataloaders(target_set_amount, target_val, target_test, BATCH_SIZE1)

    # Load two-stage models
    models = [(f'{FILENAME2}_{START_YEAR}', f'{FILENAME2}_rain_{START_YEAR}.ckpt', f'{FILENAME2}_amount_{START_YEAR}.ckpt'),
              (f'{FILENAME0}', f'{FILENAME0}_rain.ckpt', f'{FILENAME0}_amount.ckpt'),
              (f'{FILENAME3}_{START_YEAR}', f'{FILENAME3}_rain_{START_YEAR}.ckpt', f'{FILENAME3}_amount_{START_YEAR}.ckpt')]
    
    for name, rain_ckpt, amount_ckpt in models:
        print_header(name)
        rain_model = load_model(os.path.join(CKPT_DIR, rain_ckpt))
        amount_model = load_model(os.path.join(CKPT_DIR, amount_ckpt))

        # Two-stage model prediction
        rain_prob_val, amount_pred_val, y_true_val = get_two_stage_prediction(rain_model, amount_model, val_loader_rain, val_loader_amount)
        rain_prob, amount_pred, y_true = get_two_stage_prediction(rain_model, amount_model, test_loader_rain, test_loader_amount)

        # Two-stage model calibration
        y_pred = np.empty((y_true.shape[0], PREDICTION_LENGTH), dtype=np.float32)
        rain_flag = np.empty((y_true.shape[0], PREDICTION_LENGTH), dtype=np.float32)
        alarm_flag = np.empty((y_true.shape[0], PREDICTION_LENGTH), dtype=np.float32)

        for i in range(PREDICTION_LENGTH):
            cali_path = os.path.join(CALI_DIR, f'calibrator_{name}_{i+1:02}h.pkl')
            if os.path.exists(cali_path):
                best_calibrator = joblib.load(cali_path)
            else:
                rain_mask = (y_true_val[:, i] > RAIN_THRESHOLD)
                tail_mask = (y_true_val[:, i] > np.quantile(y_true_val[:, i][rain_mask], 0.80))
                best_log_mae, best_power, best_calibrator = np.inf, None, None
                for power in np.arange(1.0, 4.1, 0.5):
                    calibrator = TwoStageRainCalibrator(power)
                    calibrator.fit(rain_prob_val[:, i], amount_pred_val[:, i], y_true_val[:, i])
                    y_pred_val = calibrator.predict(rain_prob_val[:, i], amount_pred_val[:, i])[0]
                    log_y_true = np.log1p(y_true_val[:, i][tail_mask])
                    log_y_pred = np.log1p(y_pred_val[tail_mask])
                    log_mae = mean_absolute_error(log_y_true, log_y_pred)
                    if log_mae < best_log_mae:
                        best_log_mae, best_power, best_calibrator = log_mae, power, calibrator
                print(f'Lead time = {i+1}h, tail_weight_power = {best_power:.1f}, validation MAE = {best_log_mae:.4f}')
                joblib.dump(best_calibrator, cali_path)
            
            y_pred[:, i], rain_flag[:, i], alarm_flag[:, i] = best_calibrator.predict(rain_prob[:, i], amount_pred[:, i])
        
        # Two-stage model evaluation
        for i in range(PREDICTION_LENGTH):
            metrics = compute_metrics(y_true, y_pred, rain_flag, alarm_flag, i, threshold, tolerance=3)
            save_experiment(os.path.join(EVAL_DIR, f'eval_{i+1:02}h.csv'), name, metrics)

if __name__ == "__main__":
    os.makedirs(CALI_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    seed_everything(42)
    main()