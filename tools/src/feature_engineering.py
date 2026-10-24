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
# Companion notebook for the *Feature engineering* slides. Each idea is written first by
# hand with NumPy and then with the scikit-learn tool that does the same job.
#
# **Data:** the Adult census income data (does a person earn more than 50K a year?),
# a sample of 5 000 people. Rows with missing values are dropped; the *Data cleaning*
# notebook shows how to fill them instead.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer, OneHotEncoder,
                                   OrdinalEncoder, PolynomialFeatures, StandardScaler)

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
workclass = adult["workclass"].astype(str).to_numpy()
categories = np.unique(workclass)
one_hot_by_hand = (workclass[:, None] == categories[None, :]).astype(int)
print(categories)
print(one_hot_by_hand[:3])

# %%
encoder = OneHotEncoder(sparse_output=False)
one_hot_library = encoder.fit_transform(workclass.reshape(-1, 1))
print("same result:", np.array_equal(one_hot_by_hand, one_hot_library))

# %% [markdown]
# ## 2. Ordinal encoding
#
# When the categories have an order, a single number per category keeps that order.

# %%
education_order = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
                   "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors",
                   "Masters", "Prof-school", "Doctorate"]
rank = {level: position for position, level in enumerate(education_order)}
education_by_hand = adult["education"].map(rank).to_numpy()

# %%
encoder = OrdinalEncoder(categories=[education_order])
education_library = encoder.fit_transform(adult[["education"]]).ravel()
print("same result:", np.array_equal(education_by_hand, education_library))

# %% [markdown]
# ## 3. Binning
#
# Equal-width bins on $[a, b]$: width $w = (b - a)/B$ and bin index $\lfloor (x - a)/w \rfloor$
# (the largest value goes in the last bin).

# %%
ages = adult["age"].to_numpy(dtype=float)
low, high = ages.min(), ages.max()
width = (high - low) / BIN_COUNT
bins_by_hand = np.minimum(np.floor((ages - low) / width), BIN_COUNT - 1)
print("bin edges:", low + width * np.arange(BIN_COUNT + 1))

# %%
binner = KBinsDiscretizer(n_bins=BIN_COUNT, encode="ordinal", strategy="uniform")
bins_library = binner.fit_transform(ages.reshape(-1, 1)).ravel()
print("same result:", np.array_equal(bins_by_hand, bins_library))

# %% [markdown]
# ## 4. Log transform
#
# `capital-gain` is zero for most people and huge for a few. $\log(1 + x)$ pulls in the long tail.

# %%
gains = adult["capital-gain"].to_numpy(dtype=float)
log_gains_by_hand = np.log(1 + gains)
log_gains_library = FunctionTransformer(np.log1p).fit_transform(gains.reshape(-1, 1)).ravel()
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
age_hours = adult[["age", "hours-per-week"]].to_numpy(dtype=float)
age, hours = age_hours[:, 0], age_hours[:, 1]
squares_by_hand = np.column_stack([age, hours, age**2, age * hours, hours**2])

# %%
squares_library = PolynomialFeatures(degree=2, include_bias=False).fit_transform(age_hours)
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

numeric_only = make_pipeline(StandardScaler(), model)
with_categories = make_pipeline(
    make_column_transformer((StandardScaler(), numeric),
                            (OneHotEncoder(handle_unknown="ignore"), categorical)),
    model)

for name, pipeline, columns in [("numeric only", numeric_only, numeric),
                                ("numeric + one-hot", with_categories, numeric + categorical)]:
    accuracy = cross_val_score(pipeline, adult[columns], target, cv=5).mean()
    print(f"{name:18} accuracy {accuracy:.3f}")

# %% [markdown]
# ## Exercises
#
# 1. Build quantile bins for `age` with `np.quantile` and compare with `KBinsDiscretizer(strategy="quantile")`.
# 2. Add the log of `capital-gain` to the model of section 6. Does the accuracy change?
# 3. Group the countries in `native-country` that appear fewer than 50 times into one value `"Other"`.
