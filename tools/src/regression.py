# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Regression
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/03-supervised/regression/regression-notebook.ipynb)
#
# Companion notebook for the *Regression* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn tool that does the same job.
#
# **Data:** the Diabetes data (442 patients, 10 measurements, a number that measures
# disease progression one year later).

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

RANDOM_STATE = 42
DEGREE = 3
ALPHA = 1.0

diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data.to_numpy(), diabetes.target.to_numpy()
bmi = diabetes.data["bmi"].to_numpy()

# %% [markdown]
# ## 1. A line through the data
#
# For one feature, least squares gives
# $\theta_1 = \dfrac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2}$ and $\theta_0 = \bar{y} - \theta_1 \bar{x}$.

# %%
x_dev = bmi - bmi.mean()                                    # x_i - x_bar
y_dev = y - y.mean()                                        # y_i - y_bar
slope = np.sum(x_dev * y_dev) / np.sum(x_dev**2)            # theta_1
intercept = y.mean() - slope * bmi.mean()                   # theta_0
print(f"by hand: y = {intercept:.1f} + {slope:.1f} x")

# %%
line = linear_model.LinearRegression().fit(bmi.reshape(-1, 1), y)
print("same result:", np.isclose(slope, line.coef_[0]), np.isclose(intercept, line.intercept_))

plt.scatter(bmi, y, s=10, alpha=0.5)
bmi_sorted = np.sort(bmi)
plt.plot(bmi_sorted, intercept + slope * bmi_sorted, color="tab:orange")
plt.xlabel("bmi (scaled)")
plt.ylabel("disease progression")
plt.show()

# %% [markdown]
# ## 2. Many features
#
# With all features, $\hat{y} = \boldsymbol{\theta}^\top \mathbf{x}$ where $\mathbf{x}$ starts with a 1 for the intercept.
# Least squares solves the normal equations $X^\top X \boldsymbol{\theta} = X^\top \mathbf{y}$. The class keeps
# $\theta_0$ in `intercept_` and the other weights in `coef_`, like scikit-learn.

# %%
class LinearRegression:
    """Least-squares linear regression solved with the normal equations."""

    def fit(self, X, y):
        X_ones = np.column_stack([np.ones(len(X)), X])
        XtX = X_ones.T @ X_ones
        Xty = X_ones.T @ y
        theta = np.linalg.solve(XtX, Xty)      # solves X^T X theta = X^T y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_

    def score(self, X, y):
        residual_sum = np.sum((y - self.predict(X)) ** 2)
        total_sum = np.sum((y - y.mean()) ** 2)
        return 1 - residual_sum / total_sum


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
by_hand = LinearRegression().fit(X_train, y_train)

# %%
model = linear_model.LinearRegression().fit(X_train, y_train)
same_coef = np.allclose(by_hand.coef_, model.coef_)
same_intercept = np.isclose(by_hand.intercept_, model.intercept_)
print("same weights:", same_coef, same_intercept)

# %% [markdown]
# ## 3. Scoring a regression
#
# $\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|$, $\quad\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$,
# $\quad R^2 = 1 - \dfrac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$. The `score` method returns $R^2$.

# %%
y_pred = by_hand.predict(X_test)
errors = y_test - y_pred
mae = np.mean(np.abs(errors))
mse = np.mean(errors**2)
r2 = by_hand.score(X_test, y_test)
library = [
    mean_absolute_error(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    r2_score(y_test, y_pred),
]
same_scores = np.allclose([mae, mse, r2], library)
print(f"MAE {mae:.1f}, MSE {mse:.1f}, R2 {r2:.3f} | same result:", same_scores)

# %% [markdown]
# ## 4. Curved patterns
#
# Polynomial regression adds the powers $x, x^2, \dots, x^p$ as new features and fits a line to them.

# %%
powers = np.column_stack([bmi**p for p in range(1, DEGREE + 1)])
curve_by_hand = LinearRegression().fit(powers, y).predict(powers)

# %%
bmi_column = bmi.reshape(-1, 1)
curve_model = make_pipeline(PolynomialFeatures(DEGREE), linear_model.LinearRegression())
curve_model.fit(bmi_column, y)
print("same result:", np.allclose(curve_by_hand, curve_model.predict(bmi_column)))

# %% [markdown]
# ## 5. Shrinking the weights
#
# Ridge regression adds a penalty $\alpha \sum_{j \ge 1} \theta_j^2$ to the squared error, which gives
# $\boldsymbol{\theta} = (X^\top X + \alpha I)^{-1} X^\top \mathbf{y}$. The intercept is not penalized, so the data
# are centered first and $\theta_0$ is recovered from the means.

# %%
class Ridge(LinearRegression):
    """Linear regression with a squared penalty on the weights."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X_c = X - X.mean(axis=0)                           # centered data
        y_c = y - y.mean()
        penalty = self.alpha * np.eye(X.shape[1])          # alpha I
        self.coef_ = np.linalg.solve(X_c.T @ X_c + penalty, X_c.T @ y_c)
        self.intercept_ = y.mean() - X.mean(axis=0) @ self.coef_
        return self


ridge_by_hand = Ridge(alpha=ALPHA).fit(X_train, y_train)

# %%
ridge = linear_model.Ridge(alpha=ALPHA).fit(X_train, y_train)
same_coef = np.allclose(ridge_by_hand.coef_, ridge.coef_)
same_intercept = np.isclose(ridge_by_hand.intercept_, ridge.intercept_)
print("same weights:", same_coef, same_intercept)
r2_least_squares = by_hand.score(X_test, y_test)
r2_ridge = ridge_by_hand.score(X_test, y_test)
print(f"test R2: least squares {r2_least_squares:.3f}, ridge {r2_ridge:.3f}")

# %% [markdown]
# ## Exercises
#
# 1. Fit the line of section 1 with `s5` instead of `bmi`. Which feature predicts better?
# 2. Compare the test $R^2$ of the one-feature line with the model of section 2.
# 3. Raise the degree in section 4 to 10 and compare training and test error.
