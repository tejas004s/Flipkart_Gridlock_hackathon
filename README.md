# 🚦 Flipkart Traffic Demand Prediction

## Competition Overview

The objective of this challenge is to predict **traffic demand** (range: 0–1) for road segments identified by a **geohash** and a **15-minute timestamp slot**.

### Dataset

| File                  | Rows   | Columns |
| --------------------- | ------ | ------- |
| train.csv             | 77,299 | 11      |
| test.csv              | 41,778 | 10      |
| sample_submission.csv | 5      | 2       |

### Problem Statement

Predict traffic demand for future road segments and time slots using historical traffic patterns, road characteristics, weather conditions, and temporal information.

### Evaluation Metric

**R² Score (Coefficient of Determination)**

---

# 🛠️ Tools & Libraries

### Language

* Python 3.10+

### Core Libraries

* **Pandas** – Data loading, preprocessing, aggregation, target encoding
* **NumPy** – Numerical computations and feature transformations
* **Scikit-learn** – Model training and evaluation

### Visualization

* **Matplotlib** – Exploratory data analysis plots
* **Seaborn** – Correlation and distribution visualizations

### Statistical Analysis

* **SciPy** – Pearson correlation analysis

### Development Environment

* Jupyter Notebook

---

# 📊 Feature Engineering

A total of **25 engineered features** were created.

## 1. Time Features (7)

Traffic demand follows strong daily patterns. The timestamp column was transformed into cyclical and categorical representations.

| Feature     | Description                          |
| ----------- | ------------------------------------ |
| hour        | Hour extracted from timestamp        |
| minute      | Minute extracted from timestamp      |
| time_of_day | Minutes since midnight               |
| sin_time    | Cyclical time encoding (sine)        |
| cos_time    | Cyclical time encoding (cosine)      |
| is_peak     | Rush-hour indicator (7–9 AM, 5–7 PM) |
| is_night    | Night-time indicator                 |

Cyclical encoding ensures that **23:45 and 00:00 remain close in feature space**.

---

## 2. Categorical Encoding (4)

| Feature           | Encoding                                          |
| ----------------- | ------------------------------------------------- |
| RoadType_enc      | Residential=0, Commercial=1, Highway=2, Unknown=3 |
| LargeVehicles_enc | Allowed=1, Otherwise=0                            |
| Landmarks_enc     | Yes=1, No=0                                       |
| Weather_enc       | Sunny=0, Rainy=1, Cloudy=2, Snowy=3, Unknown=4    |

### Missing Value Handling

* Temperature → Filled using training median
* RoadType → Filled with `"Unknown"`
* Weather → Filled with `"Unknown"`
* LargeVehicles → Treated as `"Not Allowed"`
* Landmarks → Treated as `"No"`

---

## 3. Historical Statistical Features (6)

Historical demand statistics were computed from the training data and merged back into both train and test sets.

| Feature    | Description                            |
| ---------- | -------------------------------------- |
| gh_mean    | Mean demand by geohash                 |
| gh_std     | Demand standard deviation by geohash   |
| ts_mean    | Mean demand by timestamp               |
| ts_std     | Demand standard deviation by timestamp |
| rt_mean    | Mean demand by RoadType                |
| lanes_mean | Mean demand by NumberOfLanes           |

These features capture location-level and time-level traffic behavior.

---

## 4. Geohash × Timestamp Interaction (1)

### ⭐ Strongest Feature

**gh_ts_mean**

Average demand grouped by:

```text
(geohash, timestamp)
```

This feature effectively memorizes historical traffic demand for a specific road segment at a specific time slot.

Fallback strategy:

```text
gh_ts_mean → gh_mean
```

This feature alone can nearly reconstruct the training targets.

---

## 5. Lag Feature (1)

### ⭐ Key Generalization Feature

**lag1**

Demand from **Day 48** for the same:

```text
(geohash, timestamp)
```

### Why it matters

Traffic demand is highly correlated across consecutive days.

* Pearson Correlation ≈ **0.79**
* Implied R² ≈ **0.63**

Coverage:

* Exact Day-48 match available for **88.9%** of test rows

Fallback hierarchy:

```text
lag1 → gh_ts_mean → gh_mean
```

---

## 6. Interaction Features (4)

Non-linear feature combinations designed to help tree ensembles discover complex relationships.

| Feature    | Description             |
| ---------- | ----------------------- |
| lag1_x_ts  | lag1 × ts_mean          |
| gh_x_ts    | gh_mean × ts_mean       |
| lag1_x_gh  | lag1 × gh_mean          |
| lanes_x_ts | NumberOfLanes × ts_mean |

---

# 🤖 Model

## ExtraTreesRegressor

The final model uses **Extremely Randomized Trees (ExtraTrees)** from Scikit-Learn.

### Hyperparameters

```python
ExtraTreesRegressor(
    n_estimators=500,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)
```

### Why ExtraTrees?

* Handles non-linear relationships effectively
* Learns exact lookup-style patterns
* Fast training compared to boosting methods
* Robust against overfitting through ensemble averaging
* Excellent performance on mixed numerical and categorical features

---

# 📈 Validation Strategy

A temporal validation strategy was used to mirror the competition setup.

### Train

```text
Day 48
```

### Validation

```text
Day 49
```

### Results

| Metric           | Score  |
| ---------------- | ------ |
| Validation R²    | ~0.66  |
| Full Training R² | 1.0000 |

### Note

The large gap between validation and training performance is expected because:

* `gh_ts_mean` acts as a near-perfect memorization feature on the training set.
* Generalization relies mainly on:

  * lag1
  * timestamp statistics
  * geohash statistics

---

# 🔧 Data Preprocessing

## Missing Values

| Column        | Strategy          |
| ------------- | ----------------- |
| Temperature   | Median imputation |
| RoadType      | Unknown category  |
| Weather       | Unknown category  |
| LargeVehicles | Not Allowed       |
| Landmarks     | No                |

## Leakage Prevention

For temporal validation:

* Statistical aggregations were computed using only Day-48 data.
* Validation simulated real test-time conditions.

## Prediction Constraints

Final predictions were clipped to:

```python
[0, 1]
```

to respect the valid demand range.

---

# 📁 Project Structure

```text
.
├── train.csv
├── test.csv
├── sample_submission.csv
├── flipkart_demand_prediction.ipynb
├── submission.csv
├── eda_plots.png
├── feature_importance.png
├── predictions_analysis.png
└── README.md
```

---

# 📊 Key Insights

* Traffic demand is highly dependent on both location and time.
* Historical demand from the previous day is the strongest generalization signal.
* Geohash × Timestamp interactions provide powerful lookup-based information.
* Tree ensembles effectively capture complex traffic patterns without extensive tuning.

---

# 🚀 Reproducibility

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

Run the notebook:

```bash
jupyter notebook flipkart_demand_prediction.ipynb
```

Generate predictions:

```bash
submission.csv
```

---

## Final Performance

| Metric        | Score  |
| ------------- | ------ |
| Validation R² | ~0.66  |
| Training R²   | 1.0000 |

The solution combines temporal features, historical demand statistics, lag signals, and an ExtraTrees ensemble to model traffic demand across road segments and time slots effectively.
