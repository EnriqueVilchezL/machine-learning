# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # k-means clustering
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/04-unsupervised/kmeans/kmeans-notebook.ipynb)
#
# Companion notebook for the *k-means clustering* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn class that does the same job.
#
# **Data:** 300 synthetic points in 2-D from `make_blobs`, generated around 4 centers. The labels are thrown away.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

RANDOM_STATE = 42
K = 4
K_CHOICES = range(1, 11)
MAX_STEPS = 100

X, _ = make_blobs(n_samples=300, centers=K, cluster_std=2.0, random_state=RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)
start = X[rng.choice(len(X), size=K, replace=False)]

# %% [markdown]
# ## 1. Distance to a centroid
#
# $\lVert \mathbf{x}_i - \boldsymbol{\mu}_k \rVert^2 = \sum_j (x_{ij} - \mu_{kj})^2$, for every example against every starting centroid.

# %%
def squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every row of A to every row of B."""
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)


print("distance table:", squared_distances(X, start).shape)
print("same result:", np.allclose(squared_distances(X, start), euclidean_distances(X, start, squared=True)))

# %% [markdown]
# ## 2. Assign, update, repeat
#
# Assign: $c_i = \arg\min_k \lVert \mathbf{x}_i - \boldsymbol{\mu}_k \rVert^2$. Update: $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k} \mathbf{x}_i$.
# Stop when no example changes cluster. `fit` starts from the centroids given in `init`.

# %%
class KMeans:
    """k-means: alternate assigning examples to the nearest centroid and moving each centroid to its mean."""

    def __init__(self, n_clusters: int = 8, init: np.ndarray | None = None, max_iter: int = 300):
        self.n_clusters, self.init, self.max_iter = n_clusters, init, max_iter

    def fit(self, X, y=None):
        self.cluster_centers_ = self.init.copy()
        self.labels_ = np.full(len(X), -1)
        for step in range(self.max_iter):
            labels = self.predict(X)
            if np.array_equal(labels, self.labels_):
                break
            self.labels_ = labels
            self.cluster_centers_ = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
        self.n_iter_ = step + 1
        self.inertia_ = squared_distances(X, self.cluster_centers_).min(axis=1).sum()
        return self

    def predict(self, X):
        return squared_distances(X, self.cluster_centers_).argmin(axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


by_hand = KMeans(n_clusters=K, init=start, max_iter=MAX_STEPS).fit(X)
print("stopped after", by_hand.n_iter_, "rounds")

plt.scatter(X[:, 0], X[:, 1], c=by_hand.labels_, s=15)
plt.scatter(*by_hand.cluster_centers_.T, c="red", marker="D", s=80)
plt.show()

# %% [markdown]
# scikit-learn started from the same centroids, with a single run, follows the same steps.
# Cluster numbers are arbitrary, so the labels are compared with the adjusted Rand index (1 means the same grouping),
# and the centroids through the centroid each example gets.

# %%
kmeans = cluster.KMeans(n_clusters=K, init=start, n_init=1, max_iter=MAX_STEPS, algorithm="lloyd").fit(X)
print("same grouping:", adjusted_rand_score(by_hand.labels_, kmeans.labels_) == 1.0)
print("same number of rounds:", by_hand.n_iter_ == kmeans.n_iter_)
print("same centroid for every example:",
      np.allclose(by_hand.cluster_centers_[by_hand.labels_], kmeans.cluster_centers_[kmeans.labels_]))

# %% [markdown]
# ## 3. Inertia
#
# k-means makes the within-cluster sum of squares small: $J = \sum_{k=1}^{K}\sum_{i \in C_k} \lVert \mathbf{x}_i - \boldsymbol{\mu}_k \rVert^2$.
# `fit` stores it in `inertia_`.

# %%
print(f"J = {by_hand.inertia_:.1f} | same result:", np.isclose(by_hand.inertia_, kmeans.inertia_))

# %% [markdown]
# ## 4. Choosing K: the elbow
#
# More clusters always lower $J$. Look for the $K$ after which the curve flattens.
# Each $K$ keeps the best of 10 runs with k-means++ starts.

# %%
inertias = [cluster.KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X).inertia_ for k in K_CHOICES]

plt.plot(K_CHOICES, inertias, marker="o")
plt.xlabel("number of clusters K")
plt.ylabel("inertia J")
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. Pick other random starting centroids in the constants cell. Does the final $J$ change? Why?
# 2. Multiply the first feature of `X` by 100 and run section 2 again. What happens to the clusters?
# 3. Check that the update step never increases $J$: print the inertia after every assign and every update inside `fit`.
