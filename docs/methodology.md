# Scientific Methodology & Temporal Validation Protocol

**Project:** Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction  
**Methodology Lead:** Member 2 (Feature Engineering & Validation Lead)

---

## 1. Feature Engineering Methodology

Time-series demand features must capture daily diurnal cycles, weekly workweek patterns, annual seasonality, and autoregressive persistence while rigorously preventing future data leakage.

### 1.1 Cyclical Trigonometric Transformations
Linear numerical features (e.g., Hour $\in [0, 23]$) introduce artificial boundary discontinuities (Hour 23 and Hour 0 have a numerical distance of 23, despite being adjacent). We apply trigonometric sine and cosine encodings:

$$ x_{\text{sin}} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad x_{\text{cos}} = \cos\left(\frac{2\pi \cdot t}{T}\right) $$
- **Hour of Day ($T=24$):** Encodes diurnal solar and human lifestyle patterns.
- **Day of Week ($T=7$):** Encodes weekend drop-offs.
- **Month of Year ($T=12$):** Encodes summer cooling and winter heating seasonalities.

### 1.2 Non-Leaking Autoregressive Lags
Configurable lags: $t-1, t-2, t-24, t-48, t-168$.
For direct multi-horizon forecasting of step $h$, features satisfy:
$$ \text{Lag}_k \ge h $$

### 1.3 Shifted Rolling Statistics
To predict consumption at timestamp $t$, rolling features (24h and 168h rolling mean and standard deviation) are computed strictly on the **shifted target series**:
$$ \mu_{W}(t) = \frac{1}{W} \sum_{i=1}^{W} y(t - i) $$
$$ \sigma_{W}(t) = \sqrt{\frac{1}{W-1} \sum_{i=1}^{W} (y(t - i) - \mu_{W}(t))^2} $$
The current observation $y(t)$ is NEVER included in features predicting $\hat{y}(t)$.

---

## 2. Chronological Validation Architecture

### 2.1 The Critical Fallacy of Random Splitting
Standard $k$-fold cross-validation or random sampling assumes independent and identically distributed ($i.i.d.$) samples. In time series, observations exhibit strong temporal autocorrelation. Random splitting leaks future observations into the training partition, creating artificial, non-reproducible model accuracy that collapses in production.

### 2.2 Three-Way Chronological Partitioning
```
[================ TRAIN SET ================] [== VAL SET ==] [== TEST SET ==]
2015-09-26 09:00:00 -> 2018-06-04 00:00:00    30 Days Holdout   30 Days Holdout
(23,560 Hourly Records)                       (720 Records)     (720 Records)
```
- **Validation Set (720h):** Used for early stopping, hyperparameter tuning, and Conformal Non-Conformity calibration.
- **Holdout Test Set (720h):** Completely quarantined until final model benchmarking.

---

## 3. Probabilistic Uncertainty & Quantile Regression

### 3.1 Asymmetric Pinball Loss
Standard regression optimizes $L_2$ loss for the conditional mean. Quantile regression optimizes asymmetric pinball loss for quantile $\alpha \in (0, 1)$:

$$ \rho_\alpha(y - \hat{y}) = (y - \hat{y}) \cdot (\alpha - \mathbb{I}(y < \hat{y})) $$
We train three dedicated LightGBM estimators for $\alpha = 0.10, 0.50, 0.90$.

### 3.2 Monotonicity & Quantile Crossing Elimination
Because separate models are trained for each quantile, numerical inconsistencies may cause **quantile crossing** ($\hat{q}_{0.10} > \hat{q}_{0.50}$ or $\hat{q}_{0.50} > \hat{q}_{0.90}$). At inference, row-wise monotonicity is enforced:
$$ [\hat{L}_t, \hat{M}_t, \hat{U}_t] = \text{sort}([\hat{q}_{0.10}(t), \hat{q}_{0.50}(t), \hat{q}_{0.90}(t)]) $$

### 3.3 Split Conformal Prediction Calibration
Empirical non-conformity residuals on the validation set determine the conformal adjustment $\Delta$:
$$ E_i = \max(\hat{L}_i - y_i, y_i - \hat{U}_i) $$
$$ \Delta = \text{Quantile}\left(\{E_i\}_{i=1}^{n_{\text{val}}}, \frac{\lceil(n_{\text{val}} + 1)(1 - \alpha)\rceil}{n_{\text{val}}}\right) $$
$$ \text{Calibrated Interval: } [\hat{L}_t - \Delta, \hat{U}_t + \Delta] $$
This mathematically bounds empirical coverage: $P(y_t \in \hat{C}_t) \ge 1 - \alpha$.

