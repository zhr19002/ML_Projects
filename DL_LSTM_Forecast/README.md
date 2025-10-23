## Experiment Design
This study explores various approaches for **wave height forecasting over the next 12 hours**, considering different input configurations and forecasting strategies.

### Input Configurations
- **Baseline**: Previous hour's wave height as a persistent estimate for all future hours
- **Case 1**: Past n-hour wind observations
- **Case 2**: Past n-hour wave heights
- **Case 3**: Past n-hour wind observations + Past n-hour wave heights
- **Case 4**: Past n-hour wind observations + Future 12-hour wind forecasts
- **Case 5**: Past n-hour wind observations + Future 12-hour wind forecasts + Past n-hour wave heights

### Experiments
- **Experiment 1**: Direct forecast of wave height over the next 12 hours
- **Experiment 2**: Rolling forecast with hourly updates over the next 12 hours
- **Experiment 3**: Explore different data input window lengths on model performance
- **Experiment 4**: Explore impact of wind forecast uncertainty on model performance

### Further Model Optimization
- **Hyperparameter Tuning**
- **Bidirectional LSTM**: Incorporates both past and future temporal dependencies to improve wave height forecast accuracy
