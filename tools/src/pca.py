# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Principal component analysis
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/04-unsupervised/pca/pca-notebook.ipynb)
#
# Companion notebook for the *Principal component analysis* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn class that does the same job.
#
# **Data:** the Wine data (178 wines, 13 chemical measurements, 3 cultivars). PCA does not use the cultivar.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
Q = 2

X_raw, y = load_wine(return_X_y=True)
n, d = X_raw.shape

# %% [markdown]
# ## 1. Center and standardize
#
# The features use different units, so each one becomes $\tilde{x}_{ij} = \dfrac{x_{ij} - \bar{x}_j}{s_j}$: mean 0 and spread 1.

# %%
means = X_raw.mean(axis=0)  # x̄_j
spreads = X_raw.std(axis=0)  # s_j
X = (X_raw - means) / spreads
print("same result:", np.allclose(X, StandardScaler().fit_transform(X_raw)))

# %% [markdown]
# ## 2. Covariance and eigenvectors
#
# $\mathbf{S} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ for the centered $\mathbf{X}$. The components are its eigenvectors, $\mathbf{S}\,\mathbf{w}_k = \lambda_k \mathbf{w}_k$,
# sorted from the largest eigenvalue down. `fit` stores $\bar{\mathbf{x}}$ in `mean_`, the $\mathbf{w}_k$ as the rows of `components_`
# and the $\lambda_k$ in `explained_variance_`; `transform` gives the scores and `inverse_transform` rebuilds the examples.

# %%
class PCA:
    """Principal component analysis from the eigenvectors of the covariance matrix."""

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        S = centered.T @ centered / len(X)  # covariance S = X^T X / n
        eigenvalues, W = np.linalg.eigh(S)  # S w_k = λ_k w_k, columns of W
        order = np.argsort(eigenvalues)[::-1]  # largest eigenvalue first
        order = order[: self.n_components]  # keep the first q
        self.components_ = W[:, order].T
        self.explained_variance_ = eigenvalues[order]
        self.explained_variance_ratio_ = self.explained_variance_ / eigenvalues.sum()
        return self

    def transform(self, X):
        centered = X - self.mean_
        return centered @ self.components_.T  # z_ik = w_k^T x̃_i

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return self.mean_ + Z @ self.components_  # x̂_i = x̄ + Σ_k z_ik w_k


by_hand = PCA().fit(X)

# %% [markdown]
# scikit-learn divides by $n - 1$ instead of $n$, and the sign of each direction is arbitrary.

# %%
pca = decomposition.PCA().fit(X)
sklearn_variances = pca.explained_variance_ * (n - 1) / n  # back to dividing by n
print("same variances:", np.allclose(by_hand.explained_variance_, sklearn_variances))
same_directions = np.allclose(np.abs(by_hand.components_), np.abs(pca.components_))
print("same directions (up to sign):", same_directions)

# %% [markdown]
# ## 3. Scores: the new coordinates
#
# The score of example $i$ on component $k$ is $z_{ik} = \mathbf{w}_k^\top \tilde{\mathbf{x}}_i$. The scores are uncorrelated
# and the variance of column $k$ is $\lambda_k$.

# %%
Z = by_hand.transform(X)
print("variances are the eigenvalues:", np.allclose(Z.var(axis=0), by_hand.explained_variance_))
print("same scores (up to sign):", np.allclose(np.abs(Z), np.abs(pca.transform(X))))

plt.scatter(Z[:, 0], Z[:, 1], c=y, s=15)
plt.xlabel("score on PC1")
plt.ylabel("score on PC2")
plt.show()

# %% [markdown]
# Color shows the cultivar, which PCA never saw: two scores out of 13 features already separate the three groups.
#
# ## 4. Keeping fewer components
#
# Rebuild each example from $q$ scores: $\hat{\mathbf{x}}_i = \bar{\mathbf{x}} + \sum_{k=1}^{q} z_{ik}\,\mathbf{w}_k$.
# The average squared error is the variance left out, $\sum_{k=q+1}^{d} \lambda_k$.
# The sign of each direction cancels in $z_{ik}\,\mathbf{w}_k$, so the rebuilt examples must match exactly.

# %%
by_hand_q = PCA(n_components=Q).fit(X)
Z_q = by_hand_q.transform(X)  # q scores per example
X_hat = by_hand_q.inverse_transform(Z_q)
squared_errors = ((X - X_hat) ** 2).sum(axis=1)  # ||x̃_i - x̂_i||^2
error = squared_errors.mean()
dropped = by_hand.explained_variance_[Q:].sum()  # Σ_{k > q} λ_k
print("error equals dropped variance:", np.isclose(error, dropped))

pca_q = decomposition.PCA(n_components=Q).fit(X)
X_hat_sklearn = pca_q.inverse_transform(pca_q.transform(X))
print("same reconstruction:", np.allclose(X_hat, X_hat_sklearn))

# %% [markdown]
# ## 5. How many components: explained variance
#
# $\text{PVE}_k = \lambda_k / \sum_j \lambda_j$ and $\text{CPVE}_q = \sum_{k \le q} \lambda_k / \sum_j \lambda_j$. Choose $q$ at the elbow or at a target such as 80%.

# %%
pve = by_hand.explained_variance_ratio_
print("same result:", np.allclose(pve, pca.explained_variance_ratio_))

components = np.arange(1, d + 1)
plt.bar(components, pve, label="PVE")
plt.plot(components, np.cumsum(pve), marker="o", color="black", label="cumulative")
plt.axhline(0.8, linestyle="--", color="gray")
plt.xlabel("component k")
plt.ylabel("share of variance")
plt.legend()
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. Fit the `PCA` of section 2 on `X_raw` without standardizing. Which feature dominates PC1, and why?
# 2. Find the smallest $q$ with $\text{CPVE}_q \ge 0.8$ and compare with `decomposition.PCA(n_components=0.8)`.
# 3. Look at `by_hand.components_[0]`: which three features have the largest weights (in absolute value) on PC1?
