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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# %% [markdown]
# ## 1. Distance between examples
#
# $d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_j (a_j - b_j)^2}$, for every test flower against every training flower.

# %%
def distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance from every row of A to every row of B."""
    differences = A[:, None, :] - B[None, :, :]       # a_j - b_j for every pair
    squared_sum = (differences**2).sum(axis=2)          # sum over j
    return np.sqrt(squared_sum)


distance_by_hand = distances(X_test, X_train)
print("distance table:", distance_by_hand.shape)
print("same result:", np.allclose(distance_by_hand, euclidean_distances(X_test, X_train)))

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
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        d = distances(X, self.X_train_)
        nearest = np.argsort(d, axis=1)[:, : self.n_neighbors]   # indices of the k closest
        neighbor_labels = self.y_train_[nearest]
        proba = np.zeros((len(X), len(self.classes_)))
        for c, label in enumerate(self.classes_):
            proba[:, c] = np.mean(neighbor_labels == label, axis=1)   # share of votes for class c
        return proba

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


by_hand = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)

# %%
classifier = neighbors.KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
print("same predictions:", np.array_equal(by_hand.predict(X_test), classifier.predict(X_test)))
print("same probabilities:", np.allclose(by_hand.predict_proba(X_test), classifier.predict_proba(X_test)))
accuracy_by_hand = by_hand.score(X_test, y_test)
accuracy_library = classifier.score(X_test, y_test)
print(f"test accuracy: by hand {accuracy_by_hand:.3f}, scikit-learn {accuracy_library:.3f}")

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
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        d = distances(X, self.X_train_)
        nearest = np.argsort(d, axis=1)[:, : self.n_neighbors]   # indices of the k closest
        neighbor_values = self.y_train_[nearest]
        return neighbor_values.mean(axis=1)                      # (1/k) * sum of y_i

    def score(self, X, y):
        residual_sum = np.sum((y - self.predict(X)) ** 2)
        total_sum = np.sum((y - y.mean()) ** 2)
        return 1 - residual_sum / total_sum


X_d, y_d = load_diabetes(return_X_y=True)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_d, y_d, test_size=0.3, random_state=RANDOM_STATE
)
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
cv_accuracy = []
for k in K_CHOICES:
    pipeline = make_pipeline(StandardScaler(), neighbors.KNeighborsClassifier(n_neighbors=k))
    fold_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    cv_accuracy.append(fold_scores.mean())

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
