The primary objective of this project is to **hindcast hourly significant wave heights (SWH)** from **1973 to 2004** using historical wind data. To achieve this, an **ensemble ML framework** was developed and trained on wave and wind data spanning **2004 to 2013**.

The ensemble ML framework integrates multiple attention-based LSTM models:

- **Base Model 1**: A general wave height regressor based on an LSTM network with multi-head attention.

- **Base Model 2**: A big wave-focused regressor using wave-weighted training and the same LSTM-attention architecture.

- **Base Model 3**: A classifier estimating the probability of big wave occurrences, also built on an LSTM with multi-head attention.

- **Stacked Model**: A gating network that combines the outputs of all base models to produce the final wave height estimate.

The hindcasted wave heights (1973–2004), along with observed data (2004–2013), were then used to estimate m-year return levels via **Extreme Value Analysis (EVA)**, applying the **Generalized Pareto Distribution (GPD)**.
