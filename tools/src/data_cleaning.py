# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data cleaning
#
# Companion notebook for the *Data cleaning* slides. Each idea is written first by hand
# with NumPy and then with the library tool that does the same job.
#
# **Data:** the Titanic passenger list (891 passengers).

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
Z_LIMIT = 3
IQR_FACTOR = 1.5

titanic = sns.load_dataset("titanic")[["survived", "pclass", "sex", "age", "fare", "embarked"]]
titanic.head()

# %% [markdown]
# ## 1. Find the problems
#
# Before changing anything, count what is missing.

# %%
titanic.isna().sum()

# %% [markdown]
# ## 2. Fill missing ages with the median
#
# Take the median of the known values $\tilde{x}_j$ and put it in every empty cell.

# %%
ages = titanic["age"].to_numpy()
median_age = np.median(ages[~np.isnan(ages)])
ages_by_hand = np.where(np.isnan(ages), median_age, ages)
print("median age:", median_age)

# %%
imputer = SimpleImputer(strategy="median")
ages_library = imputer.fit_transform(titanic[["age"]]).ravel()
print("same result:", np.allclose(ages_by_hand, ages_library))

# %% [markdown]
# ## 3. Flag outliers
#
# Two rules from the slides, applied to the fare:
#
# - z-score: flag $x$ when $|z| = |x - \bar{x}| / s > 3$.
# - IQR fences: flag $x$ outside $[Q_1 - 1.5\,\text{IQR},\ Q_3 + 1.5\,\text{IQR}]$ with $\text{IQR} = Q_3 - Q_1$.

# %%
fares = titanic["fare"].to_numpy()

z_by_hand = (fares - fares.mean()) / fares.std()
q1, q3 = np.percentile(fares, [25, 75])
low_fence, high_fence = q1 - IQR_FACTOR * (q3 - q1), q3 + IQR_FACTOR * (q3 - q1)

print("z-score outliers:", np.sum(np.abs(z_by_hand) > Z_LIMIT))
print("IQR outliers:", np.sum((fares < low_fence) | (fares > high_fence)))

# %%
print("same z-scores:", np.allclose(z_by_hand, stats.zscore(fares)))

# %%
fig, axis = plt.subplots(figsize=(9, 2.5))
axis.boxplot(fares, orientation="horizontal", whis=IQR_FACTOR)
axis.axvline(high_fence, color="tab:orange", linestyle="--", label="upper IQR fence")
axis.axvline(fares.mean() + Z_LIMIT * fares.std(), color="tab:red", linestyle=":", label="z = 3")
axis.set_xlabel("fare")
axis.set_yticks([])
axis.legend()
plt.show()

# %% [markdown]
# ## 4. Learn the fill value from training rows only
#
# The median is a statistic learned from data, so it must come from the training rows.
# The same value then fills the gaps in the test rows.

# %%
train, test = train_test_split(titanic, test_size=0.2, random_state=RANDOM_STATE)

train_median = np.nanmedian(train["age"])
test_ages_by_hand = test["age"].fillna(train_median).to_numpy()

# %%
imputer = SimpleImputer(strategy="median").fit(train[["age"]])
test_ages_library = imputer.transform(test[["age"]]).ravel()
print("learned median:", imputer.statistics_[0], "| same result:", np.allclose(test_ages_by_hand, test_ages_library))

# %% [markdown]
# ## Exercises
#
# 1. Fill `embarked` with its most frequent value, by hand and with `SimpleImputer(strategy="most_frequent")`.
# 2. Fill each missing age with the median age of the passenger's class (`pclass`).
# 3. Change the IQR factor from 1.5 to 3. How many fares are still flagged?
