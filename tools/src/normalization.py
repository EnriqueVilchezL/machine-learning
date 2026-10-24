# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Normalization
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/02-data-preparation/normalization/normalization-notebook.ipynb)
#
# Companion notebook for the *Normalization* slides. Each scaler is written first by hand
# with NumPy and then with the scikit-learn class that does the same job.
#
# **Data:** the Wine recognition data (178 wines, 13 chemical measurements).

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 42

wine = load_wine(as_frame=True)
features, labels = wine.data, wine.target
features.agg(["min", "max"]).T.sort_values("max", ascending=False).head()

# %% [markdown]
# `proline` reaches the thousands while most features stay below 10. Without scaling,
# distances between wines are decided almost only by `proline`.

# %% [markdown]
# ## 1. Min-max scaling
#
# $x'_{ij} = \dfrac{x_{ij} - \min_j}{\max_j - \min_j}$ maps every feature to $[0, 1]$.
# `fit` learns $\min_j$ and $\max_j$, `transform` applies the formula.

# %%
class MinMaxScaler:
    """Map each feature to [0, 1] with the minimum and maximum learned in fit."""

    def fit(self, X):
        self.data_min_ = X.min(axis=0)  # min_j
        self.data_max_ = X.max(axis=0)  # max_j
        return self

    def transform(self, X):
        data_range = self.data_max_ - self.data_min_  # max_j - min_j
        return (X - self.data_min_) / data_range

    def fit_transform(self, X):
        return self.fit(X).transform(X)


X = features.to_numpy()
min_max_by_hand = MinMaxScaler().fit_transform(X)
min_max_library = preprocessing.MinMaxScaler().fit_transform(X)
print("same result:", np.allclose(min_max_by_hand, min_max_library))

# %% [markdown]
# ## 2. Standardization
#
# $z_{ij} = \dfrac{x_{ij} - \bar{x}_j}{s_j}$ gives each feature mean 0 and standard deviation 1.

# %%
class StandardScaler:
    """Give each feature mean 0 and standard deviation 1 with the values learned in fit."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)  # x̄_j
        self.scale_ = X.std(axis=0)  # s_j
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


standard_by_hand = StandardScaler().fit_transform(X)
standard_library = preprocessing.StandardScaler().fit_transform(X)
print("same result:", np.allclose(standard_by_hand, standard_library))

# %% [markdown]
# ## 3. Robust scaling
#
# $x'_{ij} = \dfrac{x_{ij} - \tilde{x}_j}{Q_3 - Q_1}$ uses the median and the quartiles,
# so one extreme value barely moves it.

# %%
class RobustScaler:
    """Center each feature on its median and divide by its interquartile range."""

    def fit(self, X):
        q1 = np.percentile(X, 25, axis=0)
        self.center_ = np.percentile(X, 50, axis=0)  # median x̃_j
        q3 = np.percentile(X, 75, axis=0)
        self.scale_ = q3 - q1  # interquartile range
        return self

    def transform(self, X):
        return (X - self.center_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


robust_by_hand = RobustScaler().fit_transform(X)
robust_library = preprocessing.RobustScaler().fit_transform(X)
print("same result:", np.allclose(robust_by_hand, robust_library))

# %% [markdown]
# ## 4. Fit on training rows, apply to new rows
#
# The mean and standard deviation are learned from the training rows only.
# Test rows are scaled with those same numbers.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels)

scaler_by_hand = StandardScaler().fit(X_train)
scaler_library = preprocessing.StandardScaler().fit(X_train)
X_test_by_hand = scaler_by_hand.transform(X_test)
X_test_library = scaler_library.transform(X_test)
print("same result:", np.allclose(X_test_by_hand, X_test_library))

# %% [markdown]
# ## 5. Why it matters
#
# k-nearest neighbors works with distances, so it changes a lot with scaling.
# A pipeline fits the scaler on the training rows for us.

# %%
raw_knn = KNeighborsClassifier().fit(X_train, y_train)
scaled_knn = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier())
scaled_knn.fit(X_train, y_train)
raw_accuracy = raw_knn.score(X_test, y_test)
scaled_accuracy = scaled_knn.score(X_test, y_test)
print(f"raw features:    test accuracy {raw_accuracy:.3f}")
print(f"scaled features: test accuracy {scaled_accuracy:.3f}")

# %% [markdown]
# ## 6. Log transform for a long tail
#
# Some features have a few huge values, like the population of a district in the
# California housing data. $\log_{10}(1 + x)$ spreads out the small values and pulls in the big ones.

# %%
population = fetch_california_housing(as_frame=True).frame["Population"].to_numpy()

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3))
left.hist(population, bins=50)
left.set_title("population")
log_population = np.log10(1 + population)
right.hist(log_population, bins=50, color="tab:orange")
right.set_title("log10(1 + population)")
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. Repeat section 5 with min-max and robust scaling. Which one works best here?
# 2. Scale the whole data set before splitting, then split. Which numbers did the scaler learn from the test rows?
# 3. Train a decision tree on raw and on scaled features. Why does its accuracy not change?
