# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train-test split
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/02-data-preparation/train-test-split/train-test-split-notebook.ipynb)
#
# Companion notebook for the *Train-test split* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn function that does the same job.
#
# **Data:** the Breast Cancer Wisconsin data (569 tumors, 30 measurements, malignant or benign).

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
TEST_SHARE = 0.2
REPEATS = 200

X, y = load_breast_cancer(return_X_y=True)
n = len(y)
print(f"{n} tumors, {X.shape[1]} features, malignant share {np.mean(y == 0):.2f}")

# %% [markdown]
# ## 1. Why hold data out
#
# A 1-nearest-neighbor model remembers every training example, so it is always right on them.
# Only examples it never saw tell us how good it is.

# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=TEST_SHARE, random_state=RANDOM_STATE)
memorizer = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
train_accuracy = memorizer.score(X_train, y_train)
test_accuracy = memorizer.score(X_test, y_test)
print(f"training accuracy {train_accuracy:.3f}")
print(f"test accuracy     {test_accuracy:.3f}")

# %% [markdown]
# ## 2. A shuffled split
#
# Shuffle the row numbers, then put the first $n_{\text{test}} = p\,n$ rows in the test set.
# With `stratify`, each class $c$ gives $n_{\text{test},c} \approx \pi_c \, n_{\text{test}}$ rows instead (section 3).

# %%
def train_test_split(*arrays, test_size=0.25, random_state=None, stratify=None):
    """Shuffle the rows and hold out a share of them as the test set."""
    rng = np.random.default_rng(random_state)
    n = len(arrays[0])
    n_test = int(np.ceil(test_size * n))  # n_test = p n
    if stratify is None:
        shuffled = rng.permutation(n)
        test_rows = shuffled[:n_test]
    else:
        test_parts = []
        for c in np.unique(stratify):
            rows_c = np.flatnonzero(stratify == c)  # rows of class c
            share_c = np.mean(stratify == c)  # π_c
            n_test_c = round(n_test * share_c)  # n_test,c ≈ π_c n_test
            shuffled_c = rng.permutation(rows_c)
            test_parts.append(shuffled_c[:n_test_c])
        test_rows = np.concatenate(test_parts)
    train_rows = np.setdiff1d(np.arange(n), test_rows)

    parts = []
    for array in arrays:
        parts.append(array[train_rows])
        parts.append(array[test_rows])
    return parts


train_rows, test_rows = train_test_split(np.arange(n), test_size=TEST_SHARE, random_state=RANDOM_STATE)
overlap = np.intersect1d(train_rows, test_rows)
print("train/test sizes:", len(train_rows), len(test_rows), "| overlap:", len(overlap))

# %%
train_rows_library, test_rows_library = model_selection.train_test_split(
    np.arange(n), test_size=TEST_SHARE, random_state=RANDOM_STATE)
sizes_by_hand = (len(train_rows), len(test_rows))
sizes_library = (len(train_rows_library), len(test_rows_library))
print("same sizes:", sizes_by_hand == sizes_library)

# %% [markdown]
# The two functions use different random generators, so they pick different rows; the sizes and the idea are the same.
#
# ## 3. A stratified split
#
# Split each class separately so the test set keeps the class shares $\pi_c = n_c / n$.

# %%
_, _, _, y_test_by_hand = train_test_split(
    X, y, test_size=TEST_SHARE, stratify=y, random_state=RANDOM_STATE)
share_all = np.mean(y == 0)
share_test = np.mean(y_test_by_hand == 0)
print(f"malignant share: all {share_all:.3f}, stratified test by hand {share_test:.3f}")

# %%
_, _, _, y_test_library = model_selection.train_test_split(
    X, y, test_size=TEST_SHARE, stratify=y, random_state=RANDOM_STATE)
counts_by_hand = np.bincount(y_test_by_hand)
counts_library = np.bincount(y_test_library)
print("same class counts in the test set:", np.array_equal(counts_by_hand, counts_library))

# %% [markdown]
# ## 4. A test score is noisy
#
# With accuracy $a$ on $n_{\text{test}}$ examples the standard error is
# $\text{SE} = \sqrt{a(1-a)/n_{\text{test}}}$. Repeat the split many times and compare the spread with the formula.

# %%
scores = []
for seed in range(REPEATS):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SHARE, random_state=seed)
    model = KNeighborsClassifier().fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

mean_accuracy = np.mean(scores)
n_test = len(y_test)
standard_error = np.sqrt(mean_accuracy * (1 - mean_accuracy) / n_test)  # SE = sqrt(a(1-a)/n_test)
spread = np.std(scores)
print(f"spread of {REPEATS} test scores: {spread:.3f} | formula SE: {standard_error:.3f}")

plt.hist(scores, bins=15)
plt.axvline(mean_accuracy, color="tab:orange", label="mean")
plt.xlabel("test accuracy")
plt.ylabel("splits")
plt.legend()
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. Sort the rows by `y` and split without shuffling. What happens to the class shares?
# 2. Repeat section 4 with test shares 0.1 and 0.4. How does the spread change?
# 3. Split into training, validation and test (60/20/20) and use the validation set to choose `n_neighbors`.
