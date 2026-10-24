# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # k-nearest neighbors
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/03-supervised/knn/knn-notebook.ipynb)
#
# Companion notebook for the *k-nearest neighbors* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn class that does the same job.
#
# **Data:** the Iris flowers (150 flowers, 4 measurements, 3 species) and, for regression,
# the Diabetes data.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.datasets import load_diabetes, load_iris
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
K = 5
K_CHOICES = range(1, 40, 2)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

# %% [markdown]
# ## 1. Distance between examples
#
# $d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_j (a_j - b_j)^2}$, for every test flower against every training flower.

# %%
def distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance from every row of A to every row of B."""
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


print("distance table:", distances(X_test, X_train).shape)
print("same result:", np.allclose(distances(X_test, X_train), euclidean_distances(X_test, X_train)))

# %% [markdown]
# ## 2. Vote of the k nearest
#
# Take the $k$ closest training flowers and predict the species that appears most often among them.
# `fit` only stores the training data; all the work happens in `predict`.

# %%
class KNeighborsClassifier:
    """k-NN classifier: majority vote of the k nearest training examples."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train_, self.y_train_, self.classes_ = X, y, np.unique(y)
        return self

    def predict_proba(self, X):
        nearest = np.argsort(distances(X, self.X_train_), axis=1)[:, : self.n_neighbors]
        return (self.y_train_[nearest][:, :, None] == self.classes_).mean(axis=1)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


by_hand = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)

# %%
classifier = neighbors.KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
print("same predictions:", np.array_equal(by_hand.predict(X_test), classifier.predict(X_test)))
print("same probabilities:", np.allclose(by_hand.predict_proba(X_test), classifier.predict_proba(X_test)))
print(f"test accuracy: by hand {by_hand.score(X_test, y_test):.3f}, scikit-learn {classifier.score(X_test, y_test):.3f}")

# %% [markdown]
# ## 3. Average of the k nearest
#
# For a number, the prediction is the mean of the neighbors' values: $\hat{y} = \frac{1}{k}\sum_{i \in N_k(\mathbf{x})} y_i$.

# %%
class KNeighborsRegressor:
    """k-NN regressor: mean target of the k nearest training examples."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train_, self.y_train_ = X, y
        return self

    def predict(self, X):
        nearest = np.argsort(distances(X, self.X_train_), axis=1)[:, : self.n_neighbors]
        return self.y_train_[nearest].mean(axis=1)

    def score(self, X, y):
        return 1 - np.sum((y - self.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)


X_d, y_d = load_diabetes(return_X_y=True)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_d, y_d, test_size=0.3, random_state=RANDOM_STATE)
by_hand = KNeighborsRegressor(n_neighbors=K).fit(Xd_train, yd_train)

# %%
regressor = neighbors.KNeighborsRegressor(n_neighbors=K).fit(Xd_train, yd_train)
print("same predictions:", np.allclose(by_hand.predict(Xd_test), regressor.predict(Xd_test)))
print("same R2:", np.isclose(by_hand.score(Xd_test, yd_test), regressor.score(Xd_test, yd_test)))

# %% [markdown]
# ## 4. Choosing k
#
# Small $k$ follows every point, large $k$ smooths too much. Cross-validation picks a value in between.
# The features are scaled first because k-NN works with distances.

# %%
cv_accuracy = [cross_val_score(make_pipeline(StandardScaler(), neighbors.KNeighborsClassifier(n_neighbors=k)),
                               X_train, y_train, cv=5).mean()
               for k in K_CHOICES]

plt.plot(K_CHOICES, cv_accuracy, marker="o")
plt.xlabel("number of neighbors k")
plt.ylabel("cross-validated accuracy")
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. Add a Manhattan distance $\sum_j |a_j - b_j|$ to the class of section 2 and compare with `neighbors.KNeighborsClassifier(metric="manhattan")`.
# 2. Write a weighted vote with weights $1/d$ and compare with `neighbors.KNeighborsClassifier(weights="distance")`.
# 3. Find the best $k$ from section 4 and report its test accuracy once.
