## Experiment Design
This study explores various approaches for forecasting **wave height over the next 12 hours**, considering different input configurations and forecasting strategies.

### Input Configurations
- **Baseline**: Previous hour's wave height as the prediction
- **Case 1**: Past n-hour wind features
- **Case 2**: Past n-hour wave heights
- **Case 3**: Past n-hour wind features + Past n-hour wave heights
- **Case 4**: Past n-hour wind features + Future 12-hour wind features
- **Case 5**: Past n-hour wind features + Future 12-hour wind features + Past n-hour wave heights

### Experiments
- **Experiment 1**: Direct prediction of wave height over the next 12 hours
- **Experiment 2**: Rolling forecast with hourly updates across the 12-hour horizon
- **Experiment 3**: Explore different historical input window lengths on model performance
- **Experiment 4**: Explore impact of wind forecast uncertainty on model performance

### Further Model Optimization
- **Hyperparameters Tuning**
- **Bidirectional LSTM**: Incorporates both past and future temporal dependencies to improve prediction accuracy
