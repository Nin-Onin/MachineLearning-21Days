# 🧠 Machine Learning: Zero to Professional in 21 Days
### Day 3 of 21 — Random Forests, Ensembles & Cross-Validation

---

## ✅ Day 2 Recap Check

You should have these locked in:
- [ ] Can load a messy CSV, identify and fix missing values
- [ ] Know when to drop vs. impute missing data
- [ ] Can encode categorical variables (label encoding + one-hot)
- [ ] Can engineer basic features from existing columns
- [ ] Trained Logistic Regression and read a classification report
- [ ] Understand why accuracy alone is misleading

---

## 🎯 What You're Learning Today

By the end of Day 3 you will:
1. Understand **why single decision trees fail in the real world**
2. Build a **Random Forest** — the most reliable "workhorse" algorithm in industry
3. Learn **cross-validation** — how professionals ACTUALLY evaluate models
4. Understand **feature importance** at a professional level
5. Learn the **bias-variance tradeoff** — the single most important concept in all of ML

---

## 🧩 The Core Idea: Why One Tree Is Not Enough

Think about how you make important decisions. You don't ask one person. You ask many people, aggregate their opinions, and go with the majority.

Random Forest does exactly this — but with decision trees.

```
Single Decision Tree              Random Forest
━━━━━━━━━━━━━━━━━━━━              ━━━━━━━━━━━━━
One person's opinion    vs.       100 people vote
Sensitive to noise                Robust to noise
Overfits easily                   Resistant to overfitting
High variance                     Lower variance
Fast                              Slower (but worth it)
```

**How Random Forest works (the key insight):**

1. Randomly sample rows from your training data (with replacement) → **Bagging**
2. For each tree, randomly sample only a SUBSET of features at each split
3. Train 100+ trees, each on slightly different data
4. For a new prediction: every tree votes, majority wins

The randomness is intentional. By training diverse trees, errors cancel out.
This technique — combining many weak models — is called **Ensemble Learning**.

---

## 🧩 The Bias-Variance Tradeoff (The Most Important Concept in ML)

Everything in ML is a tension between two types of error:

```
BIAS (Underfitting)                    VARIANCE (Overfitting)
━━━━━━━━━━━━━━━━━━━━━                  ━━━━━━━━━━━━━━━━━━━━━
Model too simple                       Model too complex
Misses real patterns                   Memorizes noise
Bad on train AND test                  Great on train, bad on test
Example: predict house price           Example: Decision Tree
  as the average of all houses           with no depth limit

         ↑ High Bias                              ↑ High Variance
         Bad prediction for everyone              Different data = different model
```

**The Sweet Spot:**

```
Error
 ↑
 │   Total Error
 │      ╲          ╱
 │       ╲        ╱
 │  Bias² ╲      ╱ Variance
 │          ╲  ╱
 │           ╲╱  ← Sweet spot: minimum total error
 └─────────────────────────→ Model Complexity
   Simple              Complex
```

Random Forests sit near the sweet spot. They're complex enough to capture real patterns, but the averaging prevents them from memorizing noise.

---

## 🧩 New Evaluation Tool: Cross-Validation

**The problem with a single train/test split:**

You split your data once and get 82% accuracy. But what if you were lucky? What if the 20% test set happened to be easy? What if it happened to be hard?

**Cross-validation solves this:**

```
K-Fold Cross-Validation (K=5):

Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]  → Score: 81%
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]  → Score: 83%
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]  → Score: 79%
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]  → Score: 84%
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]  → Score: 82%
                                               ───────────
                              Final Score:   Mean: 81.8% ± 1.8%
```

You get a **mean and standard deviation**. That ± number tells you how stable your model is. A model with 81% ± 1% is better than one with 83% ± 8%.

---

## 🔬 Day 3 Exercise: Random Forest on the House Prices Dataset

**Time required: 90–120 minutes**
**Goal: Build a Random Forest regressor, learn cross-validation, tune hyperparameters**

Today you switch problem types: **regression** (predicting a number, not a category).

The dataset: California housing data. 20,000+ districts, predict median house value.

This is important: most of the data cleaning you did yesterday applies here too, but today the target is a number, not a class.

---

### BLOCK 1: Load and Explore (15 min)

```python
# ── Imports ─────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Load California Housing Dataset ─────────────────────────
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target   # Median house value (in $100,000s)

print("Shape:", df.shape)
print("\nFeature descriptions:")
for name, desc in zip(housing.feature_names, housing.feature_names):
    print(f"  {name}")
print(f"  MedHouseVal (TARGET) — median house value in $100,000")
print("\nFirst 5 rows:")
print(df.head())
```

```python
# ── Statistical Summary ──────────────────────────────────────
print(df.describe().round(2))

# Missing values?
print(f"\nMissing values: {df.isnull().sum().sum()}")  # Should be 0 — clean dataset

# Distribution of target variable
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['MedHouseVal'].hist(ax=axes[0], bins=50, color='#3498db', edgecolor='white')
axes[0].set_title('Distribution of House Values')
axes[0].set_xlabel('Median House Value ($100k)')
axes[0].set_ylabel('Count')

# Correlation heatmap — find what correlates with house value
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=axes[1], cmap='coolwarm',
            annot=True, fmt='.2f', linewidths=0.5)
axes[1].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()

print("\n=== Correlations with target (MedHouseVal) ===")
print(corr['MedHouseVal'].sort_values(ascending=False).to_string())
```

**What to look for in the heatmap:**
- Cells close to +1.0 or -1.0 → strong relationship with house value
- High correlation BETWEEN features → potential redundancy (multicollinearity)

---

### BLOCK 2: Baseline Model — Linear Regression (15 min)

Always build the simplest possible model FIRST. This gives you a **baseline** to beat.
If your fancy model can't beat a straight line, something is wrong.

```python
# ── Prepare data ─────────────────────────────────────────────
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Linear Regression Baseline ───────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ── Regression Metrics (NEW — you're predicting a number now) ─
def evaluate_regression(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"  RMSE:  {rmse:.4f}  (avg error in $100k units)")
    print(f"  MAE:   {mae:.4f}  (avg absolute error)")
    print(f"  R²:    {r2:.4f}  (1.0 = perfect, 0.0 = no better than mean)")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

lr_metrics = evaluate_regression("Linear Regression (Baseline)", y_test, lr_pred)
```

**Understanding the new metrics:**

| Metric | Meaning | Good value |
|--------|---------|------------|
| **RMSE** | Root Mean Squared Error — average error, penalizes large mistakes more | Lower is better |
| **MAE** | Mean Absolute Error — average of absolute errors | Lower is better |
| **R²** | How much variance the model explains (1.0 = perfect) | Closer to 1.0 |

If R² = 0.60, the model explains 60% of house price variation. The rest is noise or missing features.

---

### BLOCK 3: Random Forest — Build and Evaluate (25 min)

```python
# ── Build Random Forest ──────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=100,    # Number of trees
    max_depth=None,      # Trees can grow as needed
    random_state=42,
    n_jobs=-1            # Use all CPU cores — makes it faster
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_metrics = evaluate_regression("Random Forest", y_test, rf_pred)

# ── Compare: Baseline vs. Random Forest ─────────────────────
print("\n=== IMPROVEMENT OVER BASELINE ===")
rmse_improvement = (lr_metrics['RMSE'] - rf_metrics['RMSE']) / lr_metrics['RMSE'] * 100
r2_improvement = rf_metrics['R2'] - lr_metrics['R2']
print(f"  RMSE reduced by:   {rmse_improvement:.1f}%")
print(f"  R² improved by:    {r2_improvement:.3f}")
```

```python
# ── Visualize Predictions vs. Reality ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, pred, name, color in zip(
    axes,
    [lr_pred, rf_pred],
    ['Linear Regression', 'Random Forest'],
    ['#e74c3c', '#27ae60']
):
    ax.scatter(y_test, pred, alpha=0.3, color=color, s=10)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'k--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('True Value ($100k)')
    ax.set_ylabel('Predicted Value ($100k)')
    ax.set_title(f'{name}\nR² = {r2_score(y_test, pred):.3f}')
    ax.legend()

plt.tight_layout()
plt.show()

# The perfect model would have all dots on the diagonal line.
# Random Forest's dots cluster much tighter to the line.
```

---

### BLOCK 4: Cross-Validation — The Professional Way (20 min)

```python
# ── 5-Fold Cross-Validation ──────────────────────────────────
# Instead of one train/test split, we test 5 times

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Linear Regression CV
lr_cv_scores = cross_val_score(
    LinearRegression(), X, y,
    cv=kf, scoring='r2', n_jobs=-1
)

# Random Forest CV (fewer trees for speed in CV)
rf_cv = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_cv_scores = cross_val_score(
    rf_cv, X, y,
    cv=kf, scoring='r2', n_jobs=-1
)

print("=" * 50)
print("CROSS-VALIDATION RESULTS (5 folds)")
print("=" * 50)
print(f"\nLinear Regression:")
print(f"  Scores:  {lr_cv_scores.round(3)}")
print(f"  Mean R²: {lr_cv_scores.mean():.3f}")
print(f"  Std:     {lr_cv_scores.std():.3f}  ← stability indicator")

print(f"\nRandom Forest:")
print(f"  Scores:  {rf_cv_scores.round(3)}")
print(f"  Mean R²: {rf_cv_scores.mean():.3f}")
print(f"  Std:     {rf_cv_scores.std():.3f}  ← stability indicator")

print("\n→ Lower std = more STABLE model across different data slices")
print("→ A model with 81% ± 1% beats one with 83% ± 8% for reliability")
```

```python
# ── Visualize CV scores ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

folds = [f'Fold {i+1}' for i in range(5)]
x = np.arange(5)
width = 0.35

bars1 = ax.bar(x - width/2, lr_cv_scores, width, label='Linear Regression',
               color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, rf_cv_scores, width, label='Random Forest',
               color='#27ae60', alpha=0.8)

ax.axhline(y=lr_cv_scores.mean(), color='#e74c3c', linestyle='--', alpha=0.7)
ax.axhline(y=rf_cv_scores.mean(), color='#27ae60', linestyle='--', alpha=0.7)

ax.set_xlabel('Fold')
ax.set_ylabel('R² Score')
ax.set_title('Cross-Validation: Stability Comparison')
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
```

---

### BLOCK 5: Feature Importance + Hyperparameter Tuning (20 min)

```python
# ── Feature Importance ───────────────────────────────────────
# Random Forest tells you how much each feature contributed to predictions
# This is one of the most useful things in practical ML

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 5))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feature_importance)))
plt.barh(feature_importance['Feature'], feature_importance['Importance'],
         color=colors)
plt.title('Random Forest Feature Importance\n(How much each feature contributes)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("Feature importance ranking:")
print(feature_importance.sort_values('Importance', ascending=False).to_string(index=False))
```

```python
# ── Hyperparameter Tuning: Manual Exploration ────────────────
# Hyperparameters = settings you choose BEFORE training
# Parameters = values the model learns DURING training

# Key Random Forest hyperparameters:
#   n_estimators: number of trees (more = better but slower)
#   max_depth: max tree depth (lower = less overfitting)
#   min_samples_leaf: min samples at leaf (higher = smoother predictions)

results = []
n_trees_options = [10, 50, 100, 200]

print("Testing different numbers of trees...")
for n in n_trees_options:
    rf_temp = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_temp, X, y, cv=3, scoring='r2', n_jobs=-1)
    results.append({'n_estimators': n, 'mean_r2': scores.mean(), 'std': scores.std()})
    print(f"  {n:3d} trees → R²: {scores.mean():.4f} ± {scores.std():.4f}")

results_df = pd.DataFrame(results)

plt.figure(figsize=(8, 4))
plt.errorbar(results_df['n_estimators'], results_df['mean_r2'],
             yerr=results_df['std'], fmt='o-', color='#27ae60',
             capsize=5, linewidth=2, markersize=8)
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Mean R² (3-fold CV)')
plt.title('How Many Trees Do You Actually Need?')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nConclusion: After a certain point, more trees = diminishing returns")
print("There's a 'good enough' point — find it, don't over-engineer.")
```

```python
# ── Depth vs. Performance: Bias-Variance in Action ──────────
print("\nTesting max_depth: watching bias-variance tradeoff...")
print("(This is what the theory looks like in actual numbers)")

depth_results = []
depths = [1, 2, 3, 5, 8, 12, None]   # None = unlimited

for depth in depths:
    rf_d = RandomForestRegressor(n_estimators=50, max_depth=depth,
                                  random_state=42, n_jobs=-1)
    rf_d.fit(X_train, y_train)
    train_r2 = r2_score(y_train, rf_d.predict(X_train))
    test_r2  = r2_score(y_test,  rf_d.predict(X_test))
    depth_label = str(depth) if depth else 'None'
    depth_results.append({'depth': depth_label, 'train_r2': train_r2, 'test_r2': test_r2})
    print(f"  depth={depth_label:4s} → train R²: {train_r2:.3f} | test R²: {test_r2:.3f}")

# The pattern you'll see:
# - depth=1: both scores low (HIGH BIAS, underfitting)
# - depth=None: train score high, test score slightly lower (some variance)
# - depth=5 or 8: sweet spot for most datasets

dr = pd.DataFrame(depth_results)
plt.figure(figsize=(9, 4))
plt.plot(range(len(dr)), dr['train_r2'], 'o-', color='#3498db', label='Train R²')
plt.plot(range(len(dr)), dr['test_r2'],  'o-', color='#e74c3c', label='Test R²')
plt.xticks(range(len(dr)), dr['depth'])
plt.xlabel('max_depth')
plt.ylabel('R² Score')
plt.title('Bias-Variance Tradeoff\n(The Gap Between Train and Test is Variance)')
plt.legend()
plt.grid(alpha=0.3)
plt.axhspan(0.7, 1.0, alpha=0.05, color='green')
plt.tight_layout()
plt.show()
```

---

### BLOCK 6: Save Your Model (10 min)

```python
# ── Save a trained model so you can reuse it ────────────────
import joblib

# Save the best model
best_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X_train, y_train)

# Save to disk
joblib.dump(best_rf, 'housing_rf_model.pkl')
print("Model saved as 'housing_rf_model.pkl'")

# Load it back (simulates what happens in production)
loaded_model = joblib.load('housing_rf_model.pkl')

# Test a prediction
sample = X_test.iloc[[0]]
pred = loaded_model.predict(sample)
actual = y_test.iloc[0]

print(f"\nLoaded model prediction:  ${pred[0]*100:.0f}k")
print(f"Actual value:             ${actual*100:.0f}k")
print(f"Error:                    ${abs(pred[0]-actual)*100:.0f}k")
```

This `joblib.dump` / `joblib.load` pattern is how models go from notebook to production.

---

## 📖 New Concepts Mastered Today

| Concept | One-Line Definition |
|---------|---------------------|
| **Ensemble** | Combining multiple models whose errors average out |
| **Random Forest** | Ensemble of decision trees, each trained on random subsets |
| **Bagging** | Training each tree on a random sample of rows (with replacement) |
| **Cross-validation** | Evaluate model on K different train/test splits for reliable score |
| **Bias** | Error from a model too simple to capture real patterns |
| **Variance** | Error from a model too sensitive to specific training data |
| **Hyperparameter** | Settings YOU choose before training (n_estimators, max_depth) |
| **Feature importance** | How much each input feature contributed to predictions |
| **RMSE / MAE / R²** | Regression metrics: error magnitude and variance explained |
| **`joblib.dump`** | Saving a trained model to disk for reuse |

---

## 🧠 The 3 Algorithm Rules You Now Know

After 3 days, you can already apply a professional decision framework:

```
WHAT IS MY PROBLEM?
│
├── Predict a CATEGORY (spam/not spam, survived/died)
│   └── START WITH: Logistic Regression
│       └── IF needs more power: Random Forest Classifier
│
└── Predict a NUMBER (house price, temperature)
    └── START WITH: Linear Regression
        └── IF needs more power: Random Forest Regressor

ALWAYS:
  1. Establish baseline first (simplest model)
  2. Use cross-validation, not a single split
  3. Check feature importance to understand the model
  4. Stop adding complexity when improvement plateaus
```

---

## 🌙 Day 3 Checkpoint

- [ ] Understand why Random Forests beat single Decision Trees
- [ ] Built a Random Forest Regressor on California housing data
- [ ] Outperformed the Linear Regression baseline
- [ ] Used 5-fold cross-validation and interpreted mean ± std
- [ ] Plotted feature importance and identified top predictors
- [ ] Demonstrated the bias-variance tradeoff with depth tuning
- [ ] Saved and reloaded a model with `joblib`

---

## 📊 Your Progress After 3 Days

```
Algorithms you can build:     Decision Tree ✓  |  Logistic Regression ✓  |  Random Forest ✓
Problem types you handle:     Classification ✓  |  Regression ✓
Data skills:                  Clean ✓  |  EDA ✓  |  Encode ✓  |  Engineer features ✓
Evaluation:                   Accuracy ✓  |  Confusion Matrix ✓  |  R² ✓  |  Cross-Val ✓
Production skills:            Save/Load model ✓
```

You are already ahead of people who have been watching ML videos for months.

---

## 🔮 Day 4 Preview

**Tomorrow**: Support Vector Machines + your first look at the data that makes or breaks models.

You'll learn:
- How SVMs find the "widest street" between classes
- Why feature scaling suddenly becomes critical (and what happens when you skip it)
- `StandardScaler` and `MinMaxScaler` — when to use which
- Building a **preprocessing pipeline** so your code is production-ready

---

*Day 3 of 21 — Machine Learning Mentorship*
