import numpy as np
from config import *
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor

class RainOccurrenceCalibrator:
    def _score(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        denominator = tp + fp + fn
        return tp / denominator if denominator > 0 else 0.0

    def fit(self, rain_prob, y_true_amount):
        rain_prob = rain_prob.reshape(-1)
        y_true_amount = y_true_amount.reshape(-1)
        y_true = (y_true_amount > RAIN_THRESHOLD).astype(np.int32)

        best_score, best_tau = -np.inf, None
        for tau in np.arange(0.05, 0.96, 0.05):
            y_pred = (rain_prob >= tau).astype(np.int32)
            score = self._score(y_true, y_pred)
            if score > best_score:
                best_score, best_tau = score, tau
        self.tau = best_tau
        
        return self

    def transform(self, rain_prob):
        rain_prob = rain_prob.reshape(-1)
        return (rain_prob >= self.tau).astype(np.float32)

class RainAmountCalibrator:
    def __init__(self, tail_weight_power):
        self.tail_weight_power = tail_weight_power
    
    def fit(self, amount_pred, y_true):
        amount_pred = amount_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        rain_mask = (y_true > RAIN_THRESHOLD)
        rain_pred = np.maximum(amount_pred[rain_mask], 0.0)
        rain_true = np.maximum(y_true[rain_mask], 0.0)

        # Isotonic calibration
        iso_mask = (rain_true <= MAX_RAIN_CALIBRATION)
        self.iso = IsotonicRegression(y_min=0.0, increasing=True, out_of_bounds='clip')
        self.iso.fit(rain_pred[iso_mask], rain_true[iso_mask])

        # Weighted log-linear tail calibration
        self.tail_threshold = np.quantile(rain_pred, 0.90)
        tail_mask = (rain_pred >= self.tail_threshold)
        if np.sum(tail_mask) >= 30:
            x_tail = np.log1p(rain_pred[tail_mask])
            y_tail = np.log1p(rain_true[tail_mask])
            weights = 1.0 + np.log1p(rain_true[tail_mask]) ** self.tail_weight_power
            self.tail_model = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=500)
            self.tail_model.fit(x_tail.reshape(-1, 1), y_tail, sample_weight=weights)
        else:
            self.tail_model = None

        return self

    def transform(self, amount_pred):
        amount_pred = amount_pred.reshape(-1)
        amount_pred = np.maximum(amount_pred, 0.0)
        calibrated = self.iso.predict(amount_pred)
        if self.tail_model is not None:
            tail_mask = (amount_pred >= self.tail_threshold)
            if np.any(tail_mask):
                x_tail = np.log1p(amount_pred[tail_mask]).reshape(-1, 1)
                tail_log_pred = self.tail_model.predict(x_tail)
                tail_pred = np.expm1(tail_log_pred)
                calibrated[tail_mask] = tail_pred
        return np.maximum(calibrated, 0.0)

class TwoStageRainCalibrator:
    def __init__(self, tail_weight_power):
        self.occurrence = RainOccurrenceCalibrator()
        self.amount = RainAmountCalibrator(tail_weight_power)

    def fit(self, rain_prob, amount_pred, y_true):
        rain_prob = rain_prob.reshape(-1)
        amount_pred = amount_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        self.occurrence.fit(rain_prob, y_true)
        self.amount.fit(amount_pred, y_true)
        return self
    
    def predict(self, rain_prob, amount_pred, hard_zero=True):
        rain_prob = rain_prob.reshape(-1)
        amount_pred = amount_pred.reshape(-1)
        calibrated_amount = self.amount.transform(amount_pred)
        if hard_zero:
            rain_flag = self.occurrence.transform(rain_prob)
            y_pred = rain_flag * calibrated_amount
        else:
            y_pred = rain_prob * calibrated_amount
        return np.maximum(y_pred, 0.0).reshape(-1, PREDICTION_LENGTH)