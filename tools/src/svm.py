# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Support vector machines
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/03-supervised/svm/svm-notebook.ipynb)
#
# Companion notebook for the *Support vector machines* slides. Each idea is written first by
# hand with NumPy and then with the scikit-learn tool that does the same job.
#
# **Data:** two species of the Iris flowers (setosa and versicolor) on two petal measurements,
# and the Breast Cancer Wisconsin data for the kernel comparison.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import hinge_loss
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
GAMMA = 0.5
LARGE_C = 1000
SOFT_C = 0.1
LEARNING_RATE = 0.1
STEPS = 5000
TOLERANCE = 0.01

iris = load_iris()
two_species = iris.target < 2
X = iris.data[two_species][:, 2:]          # petal length and width
y = np.where(iris.target[two_species] == 1, 1, -1)

# %% [markdown]
# ## 1. The decision rule
#
# A linear SVM predicts $\hat{y} = \operatorname{sign}(\mathbf{w}^\top\mathbf{x} + b)$.
# After training, scikit-learn stores $\mathbf{w}$ in `coef_` and $b$ in `intercept_`.

# %%
model = svm.SVC(kernel="linear", C=LARGE_C).fit(X, y)
w, b = model.coef_[0], model.intercept_[0]

scores_by_hand = X @ w + b
print("same scores:", np.allclose(scores_by_hand, model.decision_function(X)))
print("same predictions:", np.array_equal(np.sign(scores_by_hand), model.predict(X)))

# %% [markdown]
# ## 2. The margin
#
# The street between the classes has width $2/\lVert\mathbf{w}\rVert$. The points on its edges are the support vectors.

# %%
print(f"margin width {2 / np.linalg.norm(w):.3f}, support vectors: {len(model.support_vectors_)}")

grid = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=20)
for offset, style in [(0, "-"), (-1, "--"), (1, "--")]:
    plt.plot(grid, (offset - b - w[0] * grid) / w[1], "k" + style)
plt.scatter(*model.support_vectors_.T, s=120, facecolors="none", edgecolors="k")
plt.ylim(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.show()

# %% [markdown]
# ## 3. The hinge loss
#
# The soft margin pays $\max(0,\ 1 - y_i f(\mathbf{x}_i))$ for each point inside the street or on the wrong side.

# %%
hinge_by_hand = np.mean(np.maximum(0, 1 - y * scores_by_hand))
print(f"average hinge loss {hinge_by_hand:.4f} | same result:", np.isclose(hinge_by_hand, hinge_loss(y, scores_by_hand)))

# %% [markdown]
# ## 4. Training a soft-margin SVM by hand
#
# Training minimizes $\tfrac{1}{2}\lVert\mathbf{w}\rVert^2 + C\sum_i \max(0,\ 1 - y_i f(\mathbf{x}_i))$.
# The hinge has a corner, so we take small steps against its subgradient: only the points inside the street
# or on the wrong side push $\mathbf{w}$ and $b$, and the steps shrink as training goes on.

# %%
class LinearSVC:
    """Linear soft-margin SVM trained by subgradient descent on the hinge loss."""

    def __init__(self, C: float = 1.0, learning_rate: float = 0.1, max_iter: int = 5000):
        self.C, self.learning_rate, self.max_iter = C, learning_rate, max_iter

    def fit(self, X, y):
        self.coef_, self.intercept_ = np.zeros(X.shape[1]), 0.0
        for step in range(1, self.max_iter + 1):
            inside = y * self.decision_function(X) < 1
            rate = self.learning_rate / np.sqrt(step)
            self.coef_ = self.coef_ - rate * (self.coef_ - self.C * y[inside] @ X[inside])
            self.intercept_ = self.intercept_ + rate * self.C * y[inside].sum()
        return self

    def decision_function(self, X):
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


by_hand = LinearSVC(C=SOFT_C, learning_rate=LEARNING_RATE, max_iter=STEPS).fit(X, y)

# %% [markdown]
# scikit-learn solves the same problem exactly; the steps only get close to the answer.
# So the scores are compared with a tolerance of 0.01 (`TOLERANCE`), and the predictions must match exactly.

# %%
classifier = svm.SVC(kernel="linear", C=SOFT_C).fit(X, y)
print("scores agree within tolerance:", np.allclose(by_hand.decision_function(X), classifier.decision_function(X), atol=TOLERANCE))
print("same predictions:", np.array_equal(by_hand.predict(X), classifier.predict(X)))

# %% [markdown]
# ## 5. The RBF kernel
#
# $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \lVert\mathbf{x} - \mathbf{x}'\rVert^2)$ measures similarity: 1 for the same point, near 0 for far points.

# %%
squared_distances = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
kernel_by_hand = np.exp(-GAMMA * squared_distances)
print("same result:", np.allclose(kernel_by_hand, rbf_kernel(X, gamma=GAMMA)))

# %% [markdown]
# ## 6. Linear or curved boundary
#
# On the Breast Cancer data, compare a linear SVM with an RBF SVM. Both need scaled features.

# %%
X_bc, y_bc = load_breast_cancer(return_X_y=True)
for kernel in ["linear", "rbf"]:
    accuracy = cross_val_score(make_pipeline(StandardScaler(), svm.SVC(kernel=kernel)), X_bc, y_bc, cv=5).mean()
    print(f"{kernel:6} kernel: cross-validated accuracy {accuracy:.3f}")

# %% [markdown]
# ## Exercises
#
# 1. Lower `C` in section 1 to 0.01. What happens to the margin width and the number of support vectors?
# 2. Compute by hand the distance $|\mathbf{w}^\top\mathbf{x} + b| / \lVert\mathbf{w}\rVert$ of every support vector to the line.
# 3. Try `gamma` values 0.001, 0.01 and 0.1 for the RBF SVM in section 6.
