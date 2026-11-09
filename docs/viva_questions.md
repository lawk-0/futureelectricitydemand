# Technical Viva Preparation & University Defense Guide

**Project:** Intelligent Energy Demand Forecasting and Probabilistic Peak Load Prediction  
**Academic Target:** B.Tech CSE Final Year Project Defense  
**Audience:** University Examination Committee & Faculty Assessors

---

## Part 1: Time-Series & Data Engineering (Member 1 Focus)

### Q1: What makes time-series forecasting fundamentally different from standard regression?
**Answer:** Standard tabular regression assumes observations are independent and identically distributed ($i.i.d.$). Time-series data violates this assumption due to temporal autocorrelation (today's demand depends heavily on yesterday's demand) and temporal ordering. You cannot shuffle rows, and future data cannot be used to predict past data.

### Q2: What are the duplicate timestamps in the PJM dataset and how did you resolve them?
**Answer:** The duplicate timestamps occur in November due to Daylight Saving Time (DST) clock fall-back, where clocks are set back one hour at 2:00 AM, recording two consecutive 1:00 AM observations. We resolved this transparently by grouping duplicates and computing the arithmetic mean, logging the count in our data audit trail.

### Q3: Why is silent imputation dangerous in energy forecasting?
**Answer:** Silently dropping rows breaks the regular sampling frequency required for lag alignment ($t-24$ must correspond to exactly 24 hours ago). Silently interpolating large gaps distorts the natural variance of peak spikes. Our pipeline explicitly audits every gap, logs the imputation technique, and reports a Data Integrity Score.

---

## Part 2: Feature Engineering & Leakage Prevention (Member 2 Focus)

### Q4: Why did you encode hour and month using sine and cosine transformations?
**Answer:** Numerical hour values (0 to 23) create an artificial discontinuity between 23:00 and 00:00 (distance of 23), even though they are adjacent in time. Converting to trigonometric coordinates on a unit circle ($x = \sin(2\pi t / T), y = \cos(2\pi t / T)$) ensures that 23:00 and 00:00 have minimal Euclidean distance, preserving cyclical continuity.

### Q5: How do you mathematically guarantee zero temporal data leakage in rolling features?
**Answer:** If we calculate a 24-hour rolling mean for predicting consumption at time $t$, we must shift the series by at least 1 step before computing the window:
$$ \mu_{24}(t) = \frac{1}{24} \sum_{i=1}^{24} y(t - i) $$
If the shift were 0, $y(t)$ would be included in its own feature, resulting in target leakage and artificially inflated training metrics that fail in production.

---

## Part 3: Baseline & Statistical Models (Member 3 Focus)

### Q6: Why do we need a Seasonal Naive model when we have advanced Gradient Boosting?
**Answer:** Without a benchmark, it is impossible to determine whether a machine learning model provides genuine predictive value or is merely memorizing recent patterns. Seasonal Naive ($y_{t+h} = y_{t+h-168}$) serves as the foundational domain benchmark in energy systems. If a complex neural or boosted model cannot beat Seasonal Naive, it should not be deployed.

### Q7: How does Facebook Prophet model seasonality?
**Answer:** Prophet uses a Generalized Additive Model (GAM) decomposed into trend, seasonality, holidays, and error:
$$ y(t) = g(t) + s(t) + h(t) + \epsilon_t $$
Seasonality $s(t)$ is modeled as a Fourier series sum of sine and cosine terms ($N=3$ for daily, $N=3$ for weekly). Trend changepoints are modeled using a continuous piecewise linear function with a sparse Laplace prior.

---

## Part 4: Core Machine Learning & Uncertainty (Member 4 Focus)

### Q8: What is Quantile Regression and how does it estimate prediction intervals?
**Answer:** Standard regression minimizes Mean Squared Error ($L_2$), estimating the conditional mean $E[Y|X]$. Quantile regression optimizes the asymmetric pinball loss:
$$ \rho_\alpha(u) = \max(\alpha u, (\alpha - 1) u) $$
By training separate models for $\alpha = 0.10$ and $\alpha = 0.90$, we directly estimate the 10th and 90th conditional percentiles, providing an empirical 80% prediction interval without assuming Gaussian error distributions.

### Q9: What is Quantile Crossing and how did you resolve it?
**Answer:** Because models for $\alpha = 0.10, 0.50, 0.90$ are trained independently, their predictions may occasionally cross (e.g., predicted 10th percentile > predicted 50th percentile). We resolve this at inference time by applying row-wise monotonic sorting across the quantile predictions, guaranteeing $L_t \le M_t \le U_t$.

### Q10: What is Split Conformal Prediction?
**Answer:** Nominal 80% quantile regression does not guarantee 80% empirical coverage on unseen future test sets due to finite-sample variability and distribution shift. Split Conformal Prediction computes non-conformity residuals on a separate validation set and applies an empirical correction $\Delta$ to the prediction bounds, guaranteeing finite-sample coverage: $P(y_t \in [\hat{L}_t - \Delta, \hat{U}_t + \Delta]) \ge 1 - \alpha$.

---

## Part 5: Evaluation & Peak Demand Analytics (Member 5 Focus)

### Q11: What are PICP and MPIW?
**Answer:**
- **PICP (Prediction Interval Coverage Probability):** Percentage of actual values falling inside the prediction bounds ($[L, U]$). Target: $\ge 80\%$.
- **MPIW (Mean Prediction Interval Width):** Average width between upper and lower bounds in Megawatts ($U - L$). Measures sharpness.
- A model must balance both: high coverage (PICP) with narrow interval width (MPIW).

### Q12: Why is sMAPE preferred over traditional MAPE?
**Answer:** Traditional Mean Absolute Percentage Error (MAPE) divides by $y_t$. When actual demand approaches zero or is small, MAPE explodes toward infinity and is heavily asymmetric. Symmetric MAPE (sMAPE) divides by $(|y_t| + |\hat{y}_t|)$, bounding individual percentage errors between 0% and 200%.

### Q13: Why is a statistical peak alert different from a grid overload warning?
**Answer:** Our peak detection module identifies when predicted demand exceeds historical statistical percentiles (e.g. 90th or 95th percentile). This informs grid operators of unusually high consumption. However, whether this causes a hardware grid overload depends on physical transformer ratings, transmission line thermal limits, and real-time generation capacity, which are physical electrical engineering constraints outside consumption data alone.

---

## Part 6: System Integration & Architecture (Member 6 Focus)

### Q14: How does the Streamlit dashboard interact with the ML backend?
**Answer:** The architecture is decoupled into `src/` modules and `dashboard/` UI renderers. Streamlit calls `EnergyForecastingEngine` and `PeakDemandAnalyzer` via clean Python interfaces. Expensive computations (dataset loading, trained model weights) are cached in `st.session_state` and serialized via `joblib`, ensuring responsive interactive UI updates.

