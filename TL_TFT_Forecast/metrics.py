import numpy as np
from config import *
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Compute hourly metrics
def categorical_metrics(y_true, y_flag, idx, threshold, tolerance):
    # Tolerance window of this lead time
    start = max(0, idx - tolerance)
    end = min(y_true.shape[1], idx + tolerance + 1)
    obs_match = np.any(y_true[:, start:end] > threshold, axis=1)
    pred_match = np.any(y_flag[:, start:end].astype(bool), axis=1)

    tp, fp, fn = np.sum(obs_match & pred_match), np.sum(~obs_match & pred_match), np.sum(obs_match & ~pred_match)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    return {'CSI': csi, 'POD': pod, 'FAR': far}

def hourly_metrics(y_true, y_pred, rain_flag, alarm_flag, idx, threshold, tolerance):
    # Continuous metrics
    mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
    rmse = np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))
    bias = np.mean(y_pred[:, idx] - y_true[:, idx])
    
    # Categorical metrics
    occur = categorical_metrics(y_true, rain_flag, idx, RAIN_THRESHOLD, tolerance)
    alarm = categorical_metrics(y_true, alarm_flag, idx, threshold, tolerance)

    return {'MAE': mae, 'RMSE': rmse, 'Bias': bias,
            'CSI_occur': occur['CSI'], 'POD_occur': occur['POD'], 'FAR_occur': occur['FAR'],
            'CSI_alarm': alarm['CSI'], 'POD_alarm': alarm['POD'], 'FAR_alarm': alarm['FAR']}

# Compute event metrics
def extract_events(flags, max_gap):
    indices = np.flatnonzero(flags)
    split_pts = np.flatnonzero(np.diff(indices) > max_gap + 1) + 1
    return [(int(g[0]), int(g[-1])) for g in np.split(indices, split_pts)]

def reconstruct_obs_timeline(y_true):
    timeline = np.full(y_true.shape[0] + y_true.shape[1] - 1, np.nan)
    for (i, h), val in np.ndenumerate(y_true):
        timeline[i+h] = val
    return timeline

def reconstruct_pred_timeline(y_flag):
    N, H = y_flag.shape
    timeline = np.zeros(N + H - 1, dtype=bool)
    first_issue = np.full(N + H - 1, N, dtype=int) 
    issues, leads = np.nonzero(y_flag)
    if len(issues) > 0:
        times = issues + leads
        timeline[times] = True
        np.minimum.at(first_issue, times, issues)
    first_issue[first_issue == N] = -1
    return timeline, first_issue

def match_events(obs_events, pred_events, tolerance):
    candidates = sorted(
        (max(0, os - pe, ps - oe), oi, pi)
        for oi, (os, oe) in enumerate(obs_events)
        for pi, (ps, pe) in enumerate(pred_events)
    )
    used_obs, used_pred, matches = set(), set(), []
    for dist, oi, pi in candidates:
        if dist > tolerance:
            break
        if oi not in used_obs and pi not in used_pred:
            used_obs.add(oi)
            used_pred.add(pi)
            matches.append((oi, pi, dist))
    return matches

def event_metrics(y_true, y_flag, threshold, tolerance, obs_max_gap, pred_max_gap):
    # Reconstruct timeline and extract events
    obs_timeline = reconstruct_obs_timeline(y_true)
    pred_timeline, first_issue = reconstruct_pred_timeline(y_flag.astype(bool))
    obs_events = extract_events(obs_timeline > threshold, obs_max_gap)
    pred_events = extract_events(pred_timeline, pred_max_gap)

    # Match observed and predicted events
    matches = match_events(obs_events, pred_events, tolerance)
    match_by_obs = {oi: pi for oi, pi, _ in matches}
    
    records = []
    for oi, (os, oe) in enumerate(obs_events):
        pi = match_by_obs.get(oi)
        # Unmatched case (missed event)
        if pi is None:
            records.append({
                'obs_start': os, 'obs_end': oe, 'matched': False,
                'pred_start': None, 'pred_end': None, 'first_issue': None,
                'lead_hours': None, 'timely': False})
            continue
        # Matched case: calculate lead time
        ps, pe = pred_events[pi]
        t_min, t_max = max(0, os - tolerance), oe + tolerance
        rel_valid_times = [t for t in range(ps, pe + 1) if t_min <= t <= t_max]
        candidate_issues = [first_issue[t] for t in rel_valid_times if 0 <= first_issue[t] <= os]
        issue = min(candidate_issues) if candidate_issues else None
        lead = os - issue if issue is not None else None
        records.append({
            'obs_start': os, 'obs_end': oe, 'matched': True,
            'pred_start': ps, 'pred_end': pe, 'first_issue': issue,
            'lead_hours': lead, 'timely': (lead is not None and lead >= 0)})
    
    # Summarize statistical metrics
    n_obs, n_pred, n_match = len(obs_events), len(pred_events), len(matches)
    n_missed, n_false = n_obs - n_match, n_pred - n_match
    safe_div = lambda a, b: a / b if b else np.nan
    n_timely = sum(r['timely'] for r in records)
    leads = [r['lead_hours'] for r in records if r['lead_hours'] is not None]
    return {'event_POD': safe_div(n_match, n_obs),
            'event_FAR': safe_div(n_false, n_pred),
            'event_CSI': safe_div(n_match, n_match + n_missed + n_false),
            'timely_event_POD': safe_div(n_timely, n_obs),
            'observed_events': n_obs,
            'predicted_events': n_pred,
            'matched_events': n_match,
            'missed_events': n_missed,
            'false_events': n_false,
            'timely_events': n_timely,
            'mean_lead_hours': float(np.mean(leads)) if leads else np.nan,
    }