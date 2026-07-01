# 🧠 Machine Learning: Zero to Professional in 21 Days
### Day 5 of 21 — KNN, K-Means Clustering & Unsupervised Learning

---

## ✅ Day 4 Recap Check

- [ ] I understand WHY feature scaling breaks SVM when skipped (distance-based math)
- [ ] I can build a sklearn Pipeline that bundles scaler + model
- [ ] I know the difference between Accuracy, F1, and Recall — and when each matters
- [ ] I can read and interpret an ROC curve and AUC score
- [ ] I know when to use SVM vs. Random Forest

---

## 🎯 What You're Learning Today

Today you cross a major threshold: **unsupervised learning** — discovering patterns in data without any labels.

This covers half of real-world ML that most beginners never touch.

1. **K-Nearest Neighbors (KNN)** — your fifth supervised algorithm
2. **K-Means Clustering** — grouping unlabeled data automatically
3. **The Elbow Method** — finding the optimal number of clusters
4. **Silhouette Score** — evaluating clusters when there's no "correct answer"
5. **Customer Segmentation** — a real business project end-to-end

---

## 🧩 New Algorithm: K-Nearest Neighbors (KNN)

KNN is the most intuitive algorithm in ML:

> "To predict what class a new point belongs to, look at the K closest points in the training data. Whatever class the majority are → that's the prediction."

```
Training data (already labeled):
   ● ● ○ ○ ○
   ● ●   ○ ○
   ●   ★ ○ ○
       ↑
   New point (★). What class is it?

With K=3: find 3 nearest neighbors
   ●, ●, ○ → majority is ●  → predict ●

With K=7: find 7 nearest neighbors
   ●, ●, ●, ○, ○, ○, ○ → majority is ○  → predict ○
```

**Why K matters enormously:**
- **K=1**: Every training point is its own class. Perfect on training data, terrible on new data. Maximum overfitting.
- **K=large**: Very smooth boundaries. May underfit. Slow.
- **K=optimal**: Usually odd (to avoid ties), found via cross-validation.

**The critical detail:** KNN does NO training. It memorizes the entire training set and does all computation at prediction time. This makes it:
- ✅ Simple, easy to understand
- ✅ Naturally handles multi-class problems
- ❌ Very slow at prediction time on large datasets
- ❌ Terrible with high-dimensional data (curse of dimensionality)
- ❌ **Requires feature scaling** — same reason as SVM (distance-based)

---

## 🧩 K-Means Clustering: Learning Without Labels

Everything you've built so far required labeled data (you knew the right answers).

**K-Means works with zero labels.** You give it data, tell it how many groups you want (K), and it finds them.

**How K-Means works:**

```
Step 1: Randomly place K centroids in the data space
Step 2: Assign every point to its nearest centroid
Step 3: Move each centroid to the mean of its assigned points
Step 4: Repeat Steps 2-3 until nothing changes
```

```
Initial:     After Step 2:   After Step 3:   Converged:
  ·  ✦  ·       ·  ✦  ·       ·     ·          ·  ✦  ·
  · · ✦ ·    ●● ●✦  ○ ○    ●● ●  ○ ○ ✦      ●● ● ✦○ ○
  · · · ✦    ●● ●  ○○ ✦    ●● ●  ○○          ●● ●  ○○
```

**What K-Means is good for:**
- Customer segmentation
- Document grouping
- Image compression (grouping pixel colors)
- Anomaly detection (points far from any centroid = anomalies)

**The key limitation:** You must choose K. K-Means will happily create any K clusters you ask for, even if the data doesn't naturally have that many.

---

## 🧩 How Do You Evaluate Clustering?

Unlike supervised learning, there's no "correct answer" to check against.
Two main tools:

**1. The Elbow Method:**
Plot the sum of squared distances from each point to its centroid (called inertia) for different K values. Find where the curve "elbows" — the point of diminishing returns.

```
Inertia
 ↑
 │●
 │ ●
 │  ●
 │   ●
 │    ●─────────  ← after the elbow, adding more clusters
 │                   barely reduces inertia
 └──────────────→ K
      ↑
   Elbow here
```

**2. Silhouette Score (range: -1 to +1):**
For each point, measures how similar it is to its own cluster vs. neighboring clusters.
- `+1.0` = well-clustered, clearly belongs to its cluster
- `0.0` = on the boundary between clusters
- `-1.0` = probably in the wrong cluster

```
silhouette = (b - a) / max(a, b)
  a = average distance to points in SAME cluster
  b = average distance to points in NEAREST OTHER cluster
```

Higher silhouette = better clustering.

---

## 🔬 Day 5 Exercise: KNN + Customer Segmentation

**Time required: 90–120 minutes**

Part A (30 min): KNN on the Wine dataset
Part B (60 min): K-Means customer segmentation from scratch

---

### PART A: KNN Classification on Wine Dataset

```python
# ── Imports ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, silhouette_score,
                              silhouette_samples)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── Load Wine Dataset ─────────────────────────────────────────
# 178 wines, 13 chemical properties, 3 wine types (classes 0, 1, 2)
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print("Shape:", X.shape)
print(f"Classes: {wine.target_names}")
print(f"Class distribution:\n{y.value_counts().to_string()}")
print(f"\nFeature ranges (shows why scaling is essential):")
print(X.describe().loc[['min','max']].round(2))
```

```python
# ── Quick EDA: Are classes separable? ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Two most important features (based on prior domain knowledge)
for ax, (f1, f2) in zip(axes, [
    ('alcohol', 'flavanoids'),
    ('color_intensity', 'proline')
]):
    for i, name in enumerate(wine.target_names):
        mask = y == i
        ax.scatter(X[f1][mask], X[f2][mask], label=name, alpha=0.7, s=40)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title(f'{f1} vs {f2}')
    ax.legend()

plt.suptitle('Wine Classes — Can We Separate Them?', y=1.02)
plt.tight_layout()
plt.show()
```

```python
# ── KNN WITHOUT Scaling ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
acc_unscaled = knn_unscaled.score(X_test, y_test)
print(f"KNN (K=5) WITHOUT scaling: {acc_unscaled:.1%}")
```

```python
# ── Find Optimal K with Cross-Validation ─────────────────────
k_values = range(1, 31)
cv_scores = []
train_scores = []

pipe_base = Pipeline([('scaler', StandardScaler()),
                       ('knn', KNeighborsClassifier())])

for k in k_values:
    pipe_base.set_params(knn__n_neighbors=k)
    scores = cross_val_score(pipe_base, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

    pipe_base.fit(X_train, y_train)
    train_scores.append(pipe_base.score(X_train, y_train))

plt.figure(figsize=(10, 5))
plt.plot(k_values, train_scores, 'o-', color='#3498db',
         label='Train Accuracy', linewidth=2)
plt.plot(k_values, cv_scores, 'o-', color='#e74c3c',
         label='CV Accuracy (5-fold)', linewidth=2)
plt.xlabel('K (number of neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: Finding Optimal K\n'
          'K=1 = overfit | Large K = underfit | Sweet spot = cross-val peak')
plt.legend()
plt.grid(alpha=0.3)
plt.axvline(x=cv_scores.index(max(cv_scores))+1, color='green',
            linestyle='--', label=f'Best K={cv_scores.index(max(cv_scores))+1}')
plt.tight_layout()
plt.show()

best_k = cv_scores.index(max(cv_scores)) + 1
print(f"\nBest K = {best_k} with CV accuracy = {max(cv_scores):.1%}")
```

```python
# ── Final KNN Model with Best K ──────────────────────────────
best_knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])
best_knn_pipe.fit(X_train, y_train)
pred = best_knn_pipe.predict(X_test)

print(f"\nFinal KNN (K={best_k}) Results:")
print(classification_report(y_test, pred, target_names=wine.target_names))

print("\n=== SCALING IMPACT SUMMARY ===")
print(f"  Without scaling (K=5):              {acc_unscaled:.1%}")
print(f"  With scaling (optimal K={best_k}): {best_knn_pipe.score(X_test, y_test):.1%}")
```

---

### PART B: K-Means Customer Segmentation (Real Business Project)

This is the most practically valuable exercise this week. Companies pay data scientists significant salaries to do exactly this.

**Scenario**: You work at an e-commerce company. The marketing team has customer data — annual income and spending score — and wants to know: "Who are our distinct customer groups so we can target them differently?"

```python
# ── Create realistic customer dataset ────────────────────────
# Simulating a real retail customer dataset
np.random.seed(42)
n = 300

# Manually create 5 realistic customer segments
segments = {
    'Budget Savers':       {'income': (20, 35),  'spending': (5, 30),   'n': 60},
    'Careful Earners':     {'income': (55, 75),  'spending': (10, 35),  'n': 60},
    'Impulse Buyers':      {'income': (15, 40),  'spending': (65, 95),  'n': 60},
    'Premium Loyalists':   {'income': (70, 100), 'spending': (70, 99),  'n': 60},
    'Average Customers':   {'income': (40, 70),  'spending': (40, 65),  'n': 60},
}

data_parts = []
true_labels = []
for i, (seg_name, params) in enumerate(segments.items()):
    inc = np.random.uniform(*params['income'], params['n'])
    spen = np.random.uniform(*params['spending'], params['n'])
    data_parts.append(np.column_stack([inc, spen]))
    true_labels.extend([i] * params['n'])

customer_data = np.vstack(data_parts)
df_customers = pd.DataFrame(customer_data,
                             columns=['Annual_Income_k', 'Spending_Score'])
df_customers['CustomerID'] = range(1, len(df_customers) + 1)

print("Customer dataset shape:", df_customers.shape)
print(df_customers.describe().round(1))
```

```python
# ── Visualize raw data (no labels yet) ───────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(df_customers['Annual_Income_k'],
            df_customers['Spending_Score'],
            alpha=0.5, color='#7f8c8d', s=30)
plt.xlabel('Annual Income ($k)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Data — Before Clustering\n'
          '(Can you see natural groups?)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

```python
# ── The Elbow Method: Find the Right K ──────────────────────
X_cust = df_customers[['Annual_Income_k', 'Spending_Score']].values
scaler = StandardScaler()
X_cust_scaled = scaler.fit_transform(X_cust)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cust_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_cust_scaled, km.labels_)
    silhouette_scores.append(sil)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(K_range, inertias, 'o-', color='#e74c3c', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (within-cluster sum of squares)')
axes[0].set_title('Elbow Method\n(Look for the "bend")')
axes[0].grid(alpha=0.3)

# Silhouette plot
axes[1].plot(K_range, silhouette_scores, 'o-', color='#27ae60', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score\n(Higher = better-defined clusters)')
axes[1].grid(alpha=0.3)

best_k_sil = list(K_range)[silhouette_scores.index(max(silhouette_scores))]
axes[1].axvline(x=best_k_sil, color='red', linestyle='--',
                label=f'Best K={best_k_sil}')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\nElbow method suggests: look for the bend")
print(f"Silhouette method suggests: K = {best_k_sil} "
      f"(score = {max(silhouette_scores):.3f})")
```

```python
# ── Fit Final K-Means Model ───────────────────────────────────
optimal_k = best_k_sil  # use silhouette recommendation

km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
km_final.fit(X_cust_scaled)

df_customers['Cluster'] = km_final.labels_

# ── Visualize the Discovered Segments ────────────────────────
colors_palette = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6',
                   '#1abc9c', '#e67e22', '#34495e']
cluster_names = [f'Segment {i}' for i in range(optimal_k)]

fig, ax = plt.subplots(figsize=(10, 7))
for i in range(optimal_k):
    mask = df_customers['Cluster'] == i
    ax.scatter(df_customers[mask]['Annual_Income_k'],
               df_customers[mask]['Spending_Score'],
               color=colors_palette[i], label=cluster_names[i],
               alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

# Plot centroids (un-scaled back to original scale)
centroids_original = scaler.inverse_transform(km_final.cluster_centers_)
ax.scatter(centroids_original[:, 0], centroids_original[:, 1],
           s=250, marker='★', color='black', zorder=5, label='Centroids')

ax.set_xlabel('Annual Income ($k)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_title(f'Customer Segmentation — {optimal_k} Discovered Segments\n'
             '(No labels were used — the model found these groups)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

```python
# ── Profile each segment — the business deliverable ─────────
print("=" * 60)
print("CUSTOMER SEGMENT PROFILES")
print("=" * 60)

segment_profile = df_customers.groupby('Cluster').agg(
    Count=('CustomerID', 'count'),
    Avg_Income=('Annual_Income_k', 'mean'),
    Avg_Spending=('Spending_Score', 'mean')
).round(1)

# Add business interpretation
business_names = {
    0: 'To be determined',
    1: 'To be determined',
    2: 'To be determined',
    3: 'To be determined',
    4: 'To be determined',
}

print(segment_profile)
print()
print("Business Interpretation Guide:")
print("  High income + High spending  → Premium / VIP customers")
print("  High income + Low spending   → Careful / Conservative earners")
print("  Low income  + High spending  → Impulsive / Credit-reliant buyers")
print("  Low income  + Low spending   → Budget-conscious shoppers")
print("  Mid income  + Mid spending   → Average / Typical customers")
print()
print("Marketing actions per segment:")
print("  Premium:      Loyalty programs, exclusive early access")
print("  Careful:      Value demonstration, ROI messaging")
print("  Impulsive:    Flash sales, urgency campaigns")
print("  Budget:       Discounts, bulk-buy deals")
print("  Average:      Standard campaigns, upselling")
```

```python
# ── Predict segment for a NEW customer ───────────────────────
# This is how you'd use clustering in production

def predict_customer_segment(income, spending, model, scaler, profile):
    new_customer = np.array([[income, spending]])
    new_scaled = scaler.transform(new_customer)
    cluster = model.predict(new_scaled)[0]
    avg_income = profile.loc[cluster, 'Avg_Income']
    avg_spending = profile.loc[cluster, 'Avg_Spending']
    print(f"New customer: ${income}k income, {spending} spending score")
    print(f"  → Assigned to Segment {cluster}")
    print(f"  → Segment avg: ${avg_income:.0f}k income, {avg_spending:.0f} spending")
    return cluster

print("=== PREDICTING NEW CUSTOMER SEGMENTS ===\n")
predict_customer_segment(85, 90, km_final, scaler, segment_profile)
print()
predict_customer_segment(25, 15, km_final, scaler, segment_profile)
print()
predict_customer_segment(60, 50, km_final, scaler, segment_profile)
```

---

## 📖 New Concepts Mastered Today

| Concept | One-Line Definition |
|---------|---------------------|
| **KNN** | Classifies by majority vote among K nearest training points |
| **Lazy learning** | KNN does no training — all computation happens at prediction time |
| **Curse of dimensionality** | In high dimensions, "nearest" neighbors become meaninglessly far — KNN breaks |
| **K-Means** | Groups unlabeled data into K clusters by minimizing within-cluster distances |
| **Centroid** | The mean position of all points in a cluster — K-Means moves these iteratively |
| **Inertia** | Sum of squared distances from points to their centroid — lower = tighter clusters |
| **Elbow Method** | Plot inertia vs. K; the "elbow" bend suggests the optimal K |
| **Silhouette Score** | Measures how well each point fits its cluster vs. neighboring clusters (-1 to +1) |
| **Unsupervised Learning** | Finding patterns in data without labels — no "correct answers" to learn from |
| **Customer Segmentation** | Using clustering to group customers by behavior for targeted marketing |

---

## 🧠 Supervised vs. Unsupervised: The Full Picture So Far

```
SUPERVISED (you have labels)           UNSUPERVISED (no labels)
─────────────────────────────          ──────────────────────────
Decision Tree          ✓               K-Means Clustering      ✓
Logistic Regression    ✓               (PCA — coming Day 8)
Random Forest          ✓
SVM                    ✓
KNN                    ✓
```

The metric changes too:
- Supervised: accuracy, F1, AUC, RMSE — compare to known truth
- Unsupervised: inertia, silhouette — measure internal structure quality

---

## 🌙 Day 5 Checkpoint

- [ ] Built KNN with and without scaling — saw the performance gap
- [ ] Used cross-validation to find the optimal K
- [ ] Understand why K=1 overfits and large K underfits
- [ ] Applied K-Means to unlabeled customer data
- [ ] Used the Elbow Method AND Silhouette Score to pick K
- [ ] Visualized customer segments and profiled each one
- [ ] Predicted which segment a new customer belongs to
- [ ] Can explain the difference between supervised and unsupervised learning

---

## 🔮 Day 6 Preview

**Tomorrow**: End of Week 1 — The Full ML Project Sprint.

You'll take everything from Days 1–5 and build a **complete, polished ML project** from raw data to final model with a clean report. This is your first portfolio piece.

The project: **Credit Card Default Prediction** — real financial data, real business stakes, real messy features. You'll select the best model, tune it, and write up findings like a professional.

---

*Day 5 of 21 — Machine Learning Mentorship*
