# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Feature engineering
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/02-data-preparation/feature-engineering/feature-engineering-notebook.ipynb)
#
# Companion notebook for the *Feature engineering* slides. Each idea is written first by
# hand with NumPy and then with the scikit-learn tool that does the same job.
#
# **Data:** the Adult census income data (does a person earn more than 50K a year?),
# a sample of 5 000 people. Rows with missing values are dropped; the *Data cleaning*
# notebook shows how to fill them instead.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 42
SAMPLE_SIZE = 5_000
BIN_COUNT = 5

adult = fetch_openml("adult", version=2, as_frame=True, parser="auto").frame
adult = adult.dropna().sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
adult[["age", "workclass", "education", "hours-per-week", "capital-gain", "class"]].head()

# %% [markdown]
# ## 1. One-hot encoding
#
# A category $x$ with values $c_1, \dots, c_K$ becomes the vector
# $\mathbf{e}(x) = (\mathbb{I}(x = c_1), \dots, \mathbb{I}(x = c_K))$: one 1 and $K-1$ zeros.

# %%
class OneHotEncoder:
    """Turn each category column into one 0/1 column per category seen in fit."""

    def fit(self, X):
        self.categories_ = [np.unique(column) for column in X.T]
        return self

    def transform(self, X):
        blocks = []
        for j, categories in enumerate(self.categories_):
            column = X[:, [j]]
            block = (column == categories).astype(float)  # I(x = c_k) for every c_k
            blocks.append(block)
        return np.hstack(blocks)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


workclass = adult[["workclass"]].astype(str).to_numpy()
encoder = OneHotEncoder()
one_hot_by_hand = encoder.fit_transform(workclass)
print(encoder.categories_[0])
print(one_hot_by_hand[:3])

# %%
one_hot_library = preprocessing.OneHotEncoder(sparse_output=False).fit_transform(workclass)
print("same result:", np.array_equal(one_hot_by_hand, one_hot_library))

# %% [markdown]
# ## 2. Ordinal encoding
#
# When the categories have an order $c_1 < c_2 < \dots < c_K$, category $c_k$ becomes the single number $k - 1$.

# %%
class OrdinalEncoder:
    """Replace each category by its position in the given (or sorted) list of categories."""

    def __init__(self, categories="auto"):
        self.categories = categories

    def fit(self, X):
        self.categories_ = []
        for j, column in enumerate(X.T):
            if self.categories == "auto":
                self.categories_.append(np.unique(column))
            else:
                self.categories_.append(np.array(self.categories[j]))
        return self

    def transform(self, X):
        codes = []
        for j, categories in enumerate(self.categories_):
            column = X[:, [j]]
            position = (column == categories).argmax(axis=1)  # c_k becomes k - 1
            codes.append(position)
        return np.column_stack(codes)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


education_order = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
                   "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors",
                   "Masters", "Prof-school", "Doctorate"]
education = adult[["education"]].astype(str).to_numpy()
education_by_hand = OrdinalEncoder(categories=[education_order]).fit_transform(education)

# %%
ordinal_library = preprocessing.OrdinalEncoder(categories=[education_order])
education_library = ordinal_library.fit_transform(education)
print("same result:", np.array_equal(education_by_hand, education_library))

# %% [markdown]
# ## 3. Binning
#
# Equal-width bins on $[a, b]$: width $w = (b - a)/B$ and $\text{bin}(x) = \lfloor (x - a)/w \rfloor$
# (the largest value goes in the last bin).

# %%
class KBinsDiscretizer:
    """Cut each feature into equal-width bins between the minimum and maximum seen in fit."""

    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    def fit(self, X):
        low = X.min(axis=0)  # a
        high = X.max(axis=0)  # b
        self.bin_edges_ = np.linspace(low, high, self.n_bins + 1).T
        return self

    def transform(self, X):
        low = self.bin_edges_[:, 0]  # a
        high = self.bin_edges_[:, -1]  # b
        width = (high - low) / self.n_bins  # w = (b - a) / B
        bins = np.floor((X - low) / width)  # bin(x) = floor((x - a) / w)
        return np.clip(bins, 0, self.n_bins - 1)  # the largest value goes in the last bin

    def fit_transform(self, X):
        return self.fit(X).transform(X)


ages = adult[["age"]].to_numpy(dtype=float)
binner = KBinsDiscretizer(n_bins=BIN_COUNT)
bins_by_hand = binner.fit_transform(ages)
print("bin edges:", binner.bin_edges_[0])

# %%
binner_library = preprocessing.KBinsDiscretizer(n_bins=BIN_COUNT, encode="ordinal", strategy="uniform")
bins_library = binner_library.fit_transform(ages)
print("same result:", np.array_equal(bins_by_hand, bins_library))

# %% [markdown]
# ## 4. Log transform
#
# `capital-gain` is zero for most people and huge for a few. $\log(1 + x)$ pulls in the long tail.

# %%
gains = adult["capital-gain"].to_numpy(dtype=float)
log_gains_by_hand = np.log(1 + gains)
log_transformer = preprocessing.FunctionTransformer(np.log1p)
log_gains_library = log_transformer.fit_transform(gains.reshape(-1, 1)).ravel()
print("same result:", np.allclose(log_gains_by_hand, log_gains_library))

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3))
left.hist(gains[gains > 0], bins=30)
left.set_title("capital-gain (people with a gain)")
right.hist(log_gains_by_hand[gains > 0], bins=30, color="tab:orange")
right.set_title("log(1 + capital-gain)")
plt.show()

# %% [markdown]
# ## 5. Squares and products
#
# New columns from old ones: $x_1^2$, $x_1 x_2$ and $x_2^2$ let a linear model follow curves and interactions.

# %%
class PolynomialFeatures:
    """Add every square and every product of two columns (degree 2)."""

    def fit(self, X):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        n = self.n_features_in_
        new_columns = [X]
        for a in range(n):
            for b in range(a, n):
                new_columns.append(X[:, a] * X[:, b])  # x_a * x_b (a square when a == b)
        return np.column_stack(new_columns)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


age_hours = adult[["age", "hours-per-week"]].to_numpy(dtype=float)
squares_by_hand = PolynomialFeatures().fit_transform(age_hours)

# %%
polynomial_library = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
squares_library = polynomial_library.fit_transform(age_hours)
print("same result:", np.allclose(squares_by_hand, squares_library))

# %% [markdown]
# ## 6. Does it help?
#
# Compare a model on the raw numeric columns with one that also gets one-hot categories.

# %%
numeric = ["age", "hours-per-week", "capital-gain", "capital-loss"]
categorical = ["workclass", "education", "marital-status", "occupation"]
target = adult["class"] == ">50K"
model = LogisticRegression(max_iter=1000)

numeric_only = make_pipeline(preprocessing.StandardScaler(), model)
with_categories = make_pipeline(
    make_column_transformer((preprocessing.StandardScaler(), numeric),
                            (preprocessing.OneHotEncoder(handle_unknown="ignore"), categorical)),
    model)

for name, pipeline, columns in [("numeric only", numeric_only, numeric),
                                ("numeric + one-hot", with_categories, numeric + categorical)]:
    accuracy = cross_val_score(pipeline, adult[columns], target, cv=5).mean()
    print(f"{name:18} accuracy {accuracy:.3f}")

# %% [markdown]
# ## Exercises
#
# 1. Build quantile bins for `age` with `np.quantile` and compare with `preprocessing.KBinsDiscretizer(strategy="quantile")`.
# 2. Add the log of `capital-gain` to the model of section 6. Does the accuracy change?
# 3. Group the countries in `native-country` that appear fewer than 50 times into one value `"Other"`.
