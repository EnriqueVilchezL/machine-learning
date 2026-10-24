# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ensemble methods
#
# Companion notebook for the *Ensemble methods* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn class that does the same job.
#
# **Data:** the Breast Cancer Wisconsin data (569 tumors, 30 measurements, malignant or benign).

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TREE_COUNT = 100

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
rng = np.random.default_rng(RANDOM_STATE)
n = len(y_train)

# %% [markdown]
# ## 1. A bootstrap sample
#
# Draw $n$ rows with replacement. Each row is left out with probability $(1 - 1/n)^n \approx 0.368$.

# %%
sample = rng.integers(0, n, size=n)
left_out = 1 - len(np.unique(sample)) / n
print(f"left out: {left_out:.3f} | formula: {(1 - 1 / n) ** n:.3f}")

# %% [markdown]
# ## 2. Bagging
#
# Train one tree on each bootstrap sample and let them vote: $\hat{y} = \arg\max_c \sum_b \mathbb{I}(\hat{f}_b(\mathbf{x}) = c)$.

# %%
votes = np.zeros(len(X_test))
for _ in range(TREE_COUNT):
    sample = rng.integers(0, n, size=n)
    tree = DecisionTreeClassifier().fit(X_train[sample], y_train[sample])
    votes += tree.predict(X_test)
bagging_by_hand = (votes > TREE_COUNT / 2).astype(int)

single_tree = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=TREE_COUNT, random_state=RANDOM_STATE).fit(X_train, y_train)

print(f"one tree:            {single_tree.score(X_test, y_test):.3f}")
print(f"bagging by hand:     {np.mean(bagging_by_hand == y_test):.3f}")
print(f"bagging scikit-learn {bagging.score(X_test, y_test):.3f}")

# %% [markdown]
# The two bagging versions draw different random samples, so their accuracies are close but not identical.
#
# ## 3. Random forests
#
# A random forest is bagging plus one more twist: each split only looks at a random subset of
# about $\sqrt{d}$ features, so the trees differ more.

# %%
forest = RandomForestClassifier(n_estimators=TREE_COUNT, max_features="sqrt", random_state=RANDOM_STATE).fit(X_train, y_train)
print(f"features per split: {int(np.sqrt(X.shape[1]))} of {X.shape[1]} | test accuracy {forest.score(X_test, y_test):.3f}")

# %% [markdown]
# ## 4. Boosting, one round by hand
#
# AdaBoost trains small trees one after another. A tree with weighted error $\varepsilon$ gets the vote
# $\alpha = \frac{1}{2}\ln\frac{1 - \varepsilon}{\varepsilon}$, and the examples it got wrong weigh more in the next round.

# %%
weights = np.full(n, 1 / n)
stump = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train, sample_weight=weights)
wrong = stump.predict(X_train) != y_train
error = np.sum(weights[wrong])
alpha = 0.5 * np.log((1 - error) / error)
weights *= np.exp(np.where(wrong, alpha, -alpha))
weights /= weights.sum()
print(f"error {error:.3f}, vote alpha {alpha:.3f}, a wrong example now weighs {weights[wrong][0] / weights[~wrong][0]:.1f} times a right one")

# %%
boosting = AdaBoostClassifier(n_estimators=TREE_COUNT, random_state=RANDOM_STATE).fit(X_train, y_train)

# %% [markdown]
# ## 5. Comparing them

# %%
names = ["one tree", "bagging", "random forest", "AdaBoost"]
accuracies = [model.score(X_test, y_test) for model in [single_tree, bagging, forest, boosting]]
plt.bar(names, accuracies)
plt.ylim(0.85, 1)
plt.ylabel("test accuracy")
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. Repeat section 2 with 1, 5, 25 and 100 trees. How does the accuracy change?
# 2. Set `max_features=None` in the forest. It is now plain bagging; compare the accuracy.
# 3. Run a second round of section 4 with the new weights. Is the second stump different from the first?
