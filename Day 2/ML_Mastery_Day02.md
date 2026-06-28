# 🧠 Machine Learning: Zero to Professional in 21 Days
### Day 2 of 21 — Real Messy Data + Your Second Algorithm

---

## ✅ Day 1 Recap Check

Before starting, you should have done these yesterday:
- [ ] Built a Decision Tree on Iris and got >90% accuracy
- [ ] Demonstrated overfitting (train >> test accuracy)
- [ ] Predicted a new sample with confidence score

If you skipped Day 1's exercise, go back. Day 2 builds directly on it.

---

## 🎯 What You're Learning Today

**Yesterday**: Clean toy data. Perfect for learning structure.
**Today**: The Titanic dataset — real, historical, missing values, mixed types.

This is where 80% of beginners quit or get stuck. Not you.

### Today's Goals:
1. Learn **Exploratory Data Analysis (EDA)** — the step pros NEVER skip
2. Handle **missing values** — the #1 real-world data problem
3. **Encode categorical data** — turning words into numbers the model can use
4. Build **Logistic Regression** — your second algorithm
5. Compare two models on the SAME data — how professionals choose algorithms

---

## 🧩 New Concept: Why Data Cleaning Comes Before Modeling

Here's the professional truth that most tutorials bury:

> **In real ML work, 70–80% of your time is spent on data, not models.**

A perfect algorithm on dirty data loses to a simple algorithm on clean data. Every time.

The pipeline priority is:
```
Great Data + Simple Model   >   Dirty Data + Complex Model
```

Today you live that truth.

---

## 🧩 New Algorithm: Logistic Regression

Don't let the name confuse you. Despite saying "regression," it's a **classifier**.

**How it works (concept only — no math needed):**

Logistic Regression draws a line (or hyperplane) through your data and says:
- Everything on this side → Class A
- Everything on that side → Class B

It outputs a **probability** between 0 and 1:
- `0.85` → 85% chance this passenger survived
- `0.12` → 12% chance → predicted as "did not survive"

**When to use it:**
- Binary classification (Yes/No, Survived/Died, Spam/Not Spam)
- When you want to understand WHICH features matter most
- As your baseline before trying complex models

**Decision Tree vs Logistic Regression:**

| | Decision Tree | Logistic Regression |
|--|---------------|---------------------|
| Learns | Rules ("if petal > 2.5…") | Weights (feature importance) |
| Output | Class directly | Probability → Class |
| Overfits easily? | YES (common) | Less so |
| Interpretable? | Very (visualizable) | Yes (coefficients) |
| Best for | Non-linear patterns | Linear patterns, baselines |

---

## 🔬 Day 2 Exercise: Titanic Survival Prediction

**Time required: 90–120 minutes**
**Goal: Clean messy real data and build two models, compare them**

### The Dataset Story

April 1912. 891 passengers. Who survived?

Features available:
- `Pclass` — ticket class (1=First, 2=Second, 3=Third)
- `Sex` — male/female
- `Age` — age in years (has MISSING values)
- `SibSp` — siblings/spouses aboard
- `Parch` — parents/children aboard
- `Fare` — ticket price
- `Embarked` — port of embarkation (C, Q, S)

Target: `Survived` (0 = No, 1 = Yes)

---

### BLOCK 1: Load and First Look (15 min)

```python
# ── Import everything you need ──────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)
import warnings
warnings.filterwarnings('ignore')

# ── Load the Titanic dataset ────────────────────────────────
# pandas can load CSV directly from a URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)           # rows × columns
print("\nFirst 5 rows:")
df.head()
```

```python
# ── THE MOST IMPORTANT STEP: Know your data ─────────────────
print("=== DATA TYPES & MISSING VALUES ===")
df.info()

# Count missing values per column
print("\n=== MISSING VALUES ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_report = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
print(missing_report[missing_report['Missing'] > 0])
```

**Stop and read the output.** You'll see:
- `Age`: ~177 missing values (~20%) — manageable
- `Cabin`: ~687 missing values (~77%) — too much, we'll drop it
- `Embarked`: 2 missing — easy fix

This is your first real data problem. Now you fix it.

---

### BLOCK 2: Exploratory Data Analysis — EDA (20 min)

EDA means: **look at your data visually before touching it with a model.**

Pros do this because one visualization tells you what 1000 lines of model output can't.

```python
# ── Who survived? ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Survival by Sex
survival_by_sex = df.groupby('Sex')['Survived'].mean()
axes[0].bar(survival_by_sex.index, survival_by_sex.values, color=['#e74c3c','#3498db'])
axes[0].set_title('Survival Rate by Sex')
axes[0].set_ylabel('Survival Rate')
for i, v in enumerate(survival_by_sex.values):
    axes[0].text(i, v + 0.01, f'{v:.0%}', ha='center', fontweight='bold')

# Survival by Class
survival_by_class = df.groupby('Pclass')['Survived'].mean()
axes[1].bar(['1st Class','2nd Class','3rd Class'],
            survival_by_class.values, color=['#27ae60','#f39c12','#e74c3c'])
axes[1].set_title('Survival Rate by Passenger Class')
axes[1].set_ylabel('Survival Rate')
for i, v in enumerate(survival_by_class.values):
    axes[1].text(i, v + 0.01, f'{v:.0%}', ha='center', fontweight='bold')

# Age distribution by survival
df[df['Survived']==0]['Age'].dropna().hist(
    ax=axes[2], alpha=0.6, color='#e74c3c', label='Died', bins=20)
df[df['Survived']==1]['Age'].dropna().hist(
    ax=axes[2], alpha=0.6, color='#2ecc71', label='Survived', bins=20)
axes[2].set_title('Age Distribution by Survival')
axes[2].set_xlabel('Age')
axes[2].legend()

plt.tight_layout()
plt.show()

print("\n=== KEY INSIGHT FROM EDA ===")
print(f"Women survived at {df[df['Sex']=='female']['Survived'].mean():.0%} rate")
print(f"Men survived at   {df[df['Sex']=='male']['Survived'].mean():.0%} rate")
print(f"1st class:        {df[df['Pclass']==1]['Survived'].mean():.0%} survival rate")
print(f"3rd class:        {df[df['Pclass']==3]['Survived'].mean():.0%} survival rate")
```

**Before modeling, you now KNOW:**
- Sex is the strongest predictor
- Passenger class is a major factor
- Children had slightly higher survival

This tells you which features to prioritize. This is EDA.

---

### BLOCK 3: Data Cleaning — The Real Work (20 min)

```python
# ── Work on a copy, never the original ──────────────────────
df_clean = df.copy()

# ── FIX 1: Drop columns with too many missing values ────────
# Cabin is 77% missing — not recoverable
# Name, Ticket, PassengerId — identifiers, not predictors
df_clean = df_clean.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
print("After dropping useless columns:", df_clean.columns.tolist())

# ── FIX 2: Handle missing Age values ────────────────────────
# Strategy: fill with MEDIAN age (robust to outliers)
# DO NOT use mean — one 80-year-old skews it
median_age = df_clean['Age'].median()
df_clean['Age'].fillna(median_age, inplace=True)
print(f"\nFilled missing Age with median: {median_age}")

# ── FIX 3: Handle missing Embarked (only 2 rows) ────────────
# Strategy: fill with most common value (mode)
most_common_port = df_clean['Embarked'].mode()[0]
df_clean['Embarked'].fillna(most_common_port, inplace=True)
print(f"Filled missing Embarked with mode: '{most_common_port}'")

# ── FIX 4: Encode categorical variables ─────────────────────
# Models need NUMBERS. 'male'/'female' must become 0/1
# This is called "encoding"

df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})

# Embarked has 3 values (C, Q, S) → use one-hot encoding
# One-hot: create separate columns for each category
df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True)

print("\nFinal clean dataset:")
print(df_clean.head())
print(f"\nShape: {df_clean.shape}")
print(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
```

```python
# ── BONUS: Engineer a new feature ───────────────────────────
# Feature engineering = creating new signals from existing data
# This alone can boost model performance more than switching algorithms

df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)

print("New features added: FamilySize, IsAlone")

# Quick check: did being alone affect survival?
print(f"\nSolo travelers survived at: {df[df['SibSp']+df['Parch']==0]['Survived'].mean():.0%}")
print(f"Family travelers survived at: {df[df['SibSp']+df['Parch']>0]['Survived'].mean():.0%}")
```

---

### BLOCK 4: Build Two Models and Compare (25 min)

```python
# ── Prepare features and target ─────────────────────────────
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                'FamilySize', 'IsAlone', 'Embarked_Q', 'Embarked_S']

X = df_clean[feature_cols]
y = df_clean['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y ensures train/test have same % of survivors
)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Survival rate in train: {y_train.mean():.0%}")
print(f"Survival rate in test:  {y_test.mean():.0%}")
```

```python
# ── MODEL 1: Logistic Regression ────────────────────────────
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# ── MODEL 2: Decision Tree (your Day 1 model) ───────────────
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("=" * 45)
print(f"  Logistic Regression Accuracy:  {lr_acc:.1%}")
print(f"  Decision Tree Accuracy:        {dt_acc:.1%}")
print("=" * 45)
```

```python
# ── DEEP DIVE: What did each model get wrong? ───────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, pred, name in zip(axes,
                           [lr_pred, dt_pred],
                           ['Logistic Regression', 'Decision Tree']):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Died','Survived'],
                yticklabels=['Died','Survived'])
    ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_test, pred):.1%}')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.show()

# ── PROFESSIONAL METRIC: Classification Report ──────────────
print("=== LOGISTIC REGRESSION ===")
print(classification_report(y_test, lr_pred, target_names=['Died','Survived']))
```

---

### BLOCK 5: Understand What Your Model Learned (15 min)

```python
# ── What features matter most to Logistic Regression? ───────
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient')

plt.figure(figsize=(8, 5))
colors = ['#e74c3c' if c < 0 else '#27ae60' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.title('Feature Importance — Logistic Regression\n'
          '(Green = increases survival odds, Red = decreases)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()

print("\nTop 3 factors INCREASING survival:")
print(coef_df.tail(3)[['Feature','Coefficient']].to_string(index=False))
print("\nTop 3 factors DECREASING survival:")
print(coef_df.head(3)[['Feature','Coefficient']].to_string(index=False))
```

```python
# ── Make a real prediction ───────────────────────────────────
# You are: 30-year-old woman, 1st class, traveling alone, paid £100

you = pd.DataFrame([[1, 1, 30, 0, 0, 100.0, 1, 1, 0, 0]],
                   columns=feature_cols)

prob = lr_model.predict_proba(you)[0]
print(f"\n30-year-old woman, 1st class, traveling alone:")
print(f"  Survival probability: {prob[1]:.1%}")
print(f"  Prediction: {'SURVIVED' if prob[1] > 0.5 else 'DID NOT SURVIVE'}")

# Now try: 25-year-old man, 3rd class
him = pd.DataFrame([[3, 0, 25, 0, 0, 8.0, 1, 1, 0, 0]],
                   columns=feature_cols)
prob2 = lr_model.predict_proba(him)[0]
print(f"\n25-year-old man, 3rd class, traveling alone:")
print(f"  Survival probability: {prob2[1]:.1%}")
print(f"  Prediction: {'SURVIVED' if prob2[1] > 0.5 else 'DID NOT SURVIVE'}")
```

---

## 📖 New Concepts Mastered Today

| Concept | One-Line Definition |
|---------|---------------------|
| **EDA** | Visually exploring data BEFORE modeling to find patterns and problems |
| **Missing values** | Cells with no data — must be filled (imputed) or dropped |
| **Imputation** | Filling missing values with median, mean, or predicted value |
| **Encoding** | Converting text categories to numbers (male→0, female→1) |
| **One-hot encoding** | Turning one column with N categories into N binary columns |
| **Feature engineering** | Creating new features from existing ones (FamilySize = SibSp + Parch + 1) |
| **Stratified split** | Keeping the same class ratio in train/test split |
| **Classification report** | Precision, Recall, F1 — better than accuracy alone |
| **Coefficient** | How much each feature pushes the prediction in LR |

---

## 🧠 The Professional Insight From Today

Accuracy alone is a lie. Look at this scenario:

> If 80% of passengers died, a model that ALWAYS predicts "died" gets 80% accuracy.
> Is that a good model? No. It learned nothing.

This is why professionals use:
- **Precision**: Of everyone I predicted survived, how many actually did?
- **Recall**: Of everyone who actually survived, how many did I catch?
- **F1 Score**: The harmonic mean of precision and recall

When classes are imbalanced (many more of one class), accuracy is misleading. F1 is your friend.

---

## 🌙 Day 2 Checkpoint

- [ ] Loaded real-world messy data with pandas
- [ ] Identified missing values and chose appropriate strategies to fix them
- [ ] Performed EDA and found the top 2 survival predictors BEFORE modeling
- [ ] Encoded categorical variables (Sex, Embarked)
- [ ] Engineered 2 new features (FamilySize, IsAlone)
- [ ] Built Logistic Regression and compared it to Decision Tree
- [ ] Interpreted feature coefficients (what the model actually learned)
- [ ] Made predictions for specific hypothetical passengers

---

## 🔮 Day 3 Preview

**Tomorrow**: Random Forests — the most practically useful algorithm you'll learn this week.

You'll learn:
- Why one tree is fragile but 100 trees are robust
- The concept of **ensembles** (the secret behind most Kaggle winners)
- **Cross-validation** — a smarter way to evaluate models than a single train/test split
- How to tune a model without breaking it

---

*Day 2 of 21 — Machine Learning Mentorship*
