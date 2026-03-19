The research objective is to **improve significant wave height forecasting under wind uncertainty** using **attention-based LSTM models**. Wave observations from the WLIS buoy and wind data from Sikorsky Station (including NAM forecasts) are used to characterize wind forecast errors, which are then incorporated into model training through noise augmentation and probabilistic learning.

Three model variants are developed and evaluated:

- **Perfect-future model**: Trained with ideal wind inputs (observations treated as perfect forecasts) to assess sensitivity to wind uncertainty.

- **Noise-augmented model**: Trained with stochastic perturbations applied to wind inputs to enhance robustness against forecast errors.

- **Probabilistic model**: Extends the noise-augmented approach by predicting both mean wave height and uncertainty, using a Gaussian negative log-likelihood (NLL) loss.

This framework provides both robust deterministic predictions and probabilistic forecasts with uncertainty estimates.
