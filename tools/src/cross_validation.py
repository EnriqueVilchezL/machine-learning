# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-validation
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/02-data-preparation/cross-validation/cross-validation-notebook.ipynb)
#
# Companion notebook for the *Cross-validation* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn tool that does the same job.
#
# **Data:** the Breast Cancer Wisconsin data (569 tumors, 30 measurements, malignant or benign).

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FOLD_COUNT = 5
NEIGHBOR_CHOICES = range(1, 32, 2)

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression())

# %% [markdown]
# ## 1. k-fold cross-validation
#
# Cut the shuffled rows into $k$ folds $F_1, \dots, F_k$. Train on $k-1$ folds, score on the one
# left out, and repeat for every fold. The estimate is the average
# $\text{CV}_{(k)} = \frac{1}{k}\sum_{m} s_m$.

# %%
class KFold:
    """Cut the row numbers into k folds; each fold is the test part once."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        rows = np.arange(len(X))
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rows = rng.permutation(rows)
        for test_rows in np.array_split(rows, self.n_splits):  # folds F_1, ..., F_k
            train_rows = np.setdiff1d(rows, test_rows)  # the other k - 1 folds
            yield train_rows, test_rows


def cross_val_score(estimator, X, y, cv=5):
    """Train a fresh copy of the model on each training part and score it on the fold left out."""
    folds = KFold(cv) if isinstance(cv, int) else cv
    scores = []
    for train_rows, test_rows in folds.split(X, y):
        fold_model = clone(estimator)
        fold_model.fit(X[train_rows], y[train_rows])
        score = fold_model.score(X[test_rows], y[test_rows])  # s_m
        scores.append(score)
    return np.array(scores)


shuffled_folds = KFold(FOLD_COUNT, shuffle=True, random_state=RANDOM_STATE)
fold_scores = cross_val_score(model, X, y, cv=shuffled_folds)
cv_estimate = fold_scores.mean()  # CV_(k) = (1/k) Σ s_m
print("fold scores:", np.round(fold_scores, 3))
print(f"CV estimate {cv_estimate:.3f}, spread {fold_scores.std():.3f}")

# %% [markdown]
# Without shuffling, both `KFold` classes cut the rows in the same places:

# %%
our_folds = KFold(FOLD_COUNT).split(X)
their_folds = model_selection.KFold(FOLD_COUNT).split(X)
same_cuts = True
for (_, ours), (_, theirs) in zip(our_folds, their_folds):
    if not np.array_equal(ours, theirs):
        same_cuts = False
print("same folds:", same_cuts)

# %%
same_folds = list(KFold(FOLD_COUNT, shuffle=True, random_state=RANDOM_STATE).split(X))
library_scores = model_selection.cross_val_score(model, X, y, cv=same_folds)
print("same scores on the same folds:", np.allclose(fold_scores, library_scores))

# %% [markdown]
# In practice scikit-learn makes the folds itself:

# %%
folds = model_selection.StratifiedKFold(FOLD_COUNT, shuffle=True, random_state=RANDOM_STATE)
scores = model_selection.cross_val_score(model, X, y, cv=folds)
print(f"CV estimate {scores.mean():.3f} ± {scores.std():.3f}")

# %% [markdown]
# `StratifiedKFold` builds the folds class by class, so every fold keeps the share of malignant tumors.
#
# ## 2. Choosing a setting
#
# Try each value of a setting and keep the one with the best cross-validated score:
# $\hat{\theta} = \arg\max_\theta \text{CV}_{(k)}(\theta)$. Here the setting is the number of neighbors in k-nearest neighbors.

# %%
cv_accuracy = []
for k in NEIGHBOR_CHOICES:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    scores_k = cross_val_score(knn, X, y, cv=FOLD_COUNT)
    cv_accuracy.append(scores_k.mean())  # CV_(k)(θ)
best_index = int(np.argmax(cv_accuracy))  # arg max over θ
best_by_hand = list(NEIGHBOR_CHOICES)[best_index]
print("best number of neighbors:", best_by_hand)

plt.plot(list(NEIGHBOR_CHOICES), cv_accuracy, marker="o")
plt.xlabel("number of neighbors")
plt.ylabel("cross-validated accuracy")
plt.show()

# %%
knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = {"kneighborsclassifier__n_neighbors": list(NEIGHBOR_CHOICES)}
search = model_selection.GridSearchCV(knn, grid, cv=model_selection.KFold(FOLD_COUNT))
search.fit(X, y)
best_library = search.best_params_["kneighborsclassifier__n_neighbors"]
print("same choice:", best_library == best_by_hand)

# %% [markdown]
# ## 3. The honest protocol
#
# Keep a test set aside, choose the setting with cross-validation on the training part only,
# and score the final model once on the test set.

# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
search.fit(X_train, y_train)
test_accuracy = search.score(X_test, y_test)
print("chosen:", search.best_params_, f"| test accuracy {test_accuracy:.3f}")

# %% [markdown]
# ## Exercises
#
# 1. Repeat section 1 with $k = 2$ and $k = 10$. How do the average and the spread change?
# 2. Use `LeaveOneOut()` as `cv` in `cross_val_score`. How many models does it train?
# 3. Write the stratified folds by hand: split the rows of each class into $k$ parts and join them.
