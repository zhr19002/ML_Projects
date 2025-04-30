This project focuses on developing multiple deep learning models using the TensorFlow Keras framework to forecast wave heights based on meteorological data. The models include:
- **Base model 1**: An LSTM regression model designed to predict overall wave heights.
- **Base model 2**: An LSTM regression model tailored for forecasting big waves.
- **Base model 3**: An LSTM classification model that estimates the probability of wave height spikes.
- **Stacked model**: A gating network that integrates the outputs of the base models to produce a final prediction.
