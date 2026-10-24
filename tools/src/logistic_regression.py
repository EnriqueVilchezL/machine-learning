# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Logistic regression
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/03-supervised/logistic-regression/logistic-regression-notebook.ipynb)
#
# Companion notebook for the *Logistic regression* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn tool that does the same job.
#
# **Data:** the Breast Cancer Wisconsin data (569 tumors, 30 measurements, malignant or benign).

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
LEARNING_RATE = 0.5
STEPS = 5000

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ## 1. The model
#
# A linear score $z = \boldsymbol{\theta}^\top \mathbf{x} + \theta_0$ goes through the sigmoid
# $\sigma(z) = \dfrac{1}{1 + e^{-z}}$, which gives the probability $P(y = 1 \mid \mathbf{x})$.

# %%
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash any number into (0, 1)."""
    return 1 / (1 + np.exp(-z))


print(sigmoid(np.array([-5, 0, 2, 5])).round(3))

# %% [markdown]
# ## 2. Learning the weights by hand
#
# Gradient descent on the log-loss: $\theta_j \leftarrow \theta_j - \alpha \frac{1}{n}\sum_i (p_i - y_i)\, x_{ij}$.
# scikit-learn also adds a small penalty $\frac{1}{2Cn}\sum_{j \ge 1} \theta_j^2$ on large weights, so the class adds
# its gradient $\frac{\theta_j}{Cn}$ too, with the same default $C = 1$.

# %%
class LogisticRegression:
    """Binary logistic regression trained by gradient descent on the log-loss."""

    def __init__(self, C: float = 1.0, learning_rate: float = 0.5, n_steps: int = 5000):
        self.C = C
        self.learning_rate = learning_rate
        self.n_steps = n_steps

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n = len(y)
        self.coef_ = np.zeros(X.shape[1])                      # theta
        self.intercept_ = 0.0                                  # theta_0
        self.loss_curve_ = []
        for _ in range(self.n_steps):
            z = X @ self.coef_ + self.intercept_               # linear score
            p = sigmoid(z)                                     # P(y = 1 | x)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            self.loss_curve_.append(loss)
            error = p - y                                      # p_i - y_i
            grad_coef = X.T @ error / n                        # (1/n) sum_i (p_i - y_i) x_ij
            grad_penalty = self.coef_ / (self.C * n)           # theta_j / (C n)
            grad_intercept = np.mean(error)
            self.coef_ -= self.learning_rate * (grad_coef + grad_penalty)
            self.intercept_ -= self.learning_rate * grad_intercept
        return self

    def predict_proba(self, X):
        z = X @ self.coef_ + self.intercept_
        p = sigmoid(z)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        predicted_index = (p >= 0.5).astype(int)
        return self.classes_[predicted_index]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


by_hand = LogisticRegression(learning_rate=LEARNING_RATE, n_steps=STEPS).fit(X_train, y_train)

plt.plot(by_hand.loss_curve_)
plt.xlabel("step")
plt.ylabel("log-loss")
plt.show()

# %%
model = linear_model.LogisticRegression().fit(X_train, y_train)
print("same predictions:", np.array_equal(by_hand.predict(X_test), model.predict(X_test)))
same_proba = np.allclose(by_hand.predict_proba(X_test), model.predict_proba(X_test), atol=0.01)
print("same probabilities (to 0.01):", same_proba)
accuracy_by_hand = by_hand.score(X_test, y_test)
accuracy_library = model.score(X_test, y_test)
print(f"test accuracy: by hand {accuracy_by_hand:.3f}, scikit-learn {accuracy_library:.3f}")

# %% [markdown]
# The two solutions are close but not identical: scikit-learn uses a different solver that stops
# at its own tolerance.
#
# ## 3. The log-loss
#
# $J = -\frac{1}{n}\sum_i \big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big]$: confident wrong answers cost a lot.

# %%
p_test = model.predict_proba(X_test)[:, 1]
loss_by_hand = -np.mean(y_test * np.log(p_test) + (1 - y_test) * np.log(1 - p_test))
same_loss = np.isclose(loss_by_hand, log_loss(y_test, p_test))
print(f"log-loss {loss_by_hand:.3f} | same result:", same_loss)

# %% [markdown]
# ## 4. The probability comes from the weights
#
# scikit-learn stores $\theta_0$ in `intercept_` and $\boldsymbol{\theta}$ in `coef_`; the sigmoid of the score is `predict_proba`.

# %%
z_test = X_test @ model.coef_[0] + model.intercept_[0]
p_from_weights = sigmoid(z_test)
print("same result:", np.allclose(p_from_weights, p_test))

# %% [markdown]
# ## 5. Moving the threshold
#
# Predicting class 1 when $p \ge t$: a higher threshold makes fewer but surer positive predictions.

# %%
for threshold in [0.3, 0.5, 0.8]:
    predicted = (p_test >= threshold).astype(int)
    precision = precision_score(y_test, predicted)
    recall = recall_score(y_test, predicted)
    print(f"t = {threshold}: precision {precision:.3f}, recall {recall:.3f}")

# %% [markdown]
# ## Exercises
#
# 1. Change `LEARNING_RATE` to 0.01 and to 5. What happens to the loss curve?
# 2. Compute the odds $p/(1-p)$ for three test tumors and check that their log equals the score $z$.
# 3. Find the feature with the largest weight in `model.coef_` and explain what $e^{\theta_j}$ means for it.
