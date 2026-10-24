# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Naive Bayes
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/03-supervised/naive-bayes/naive-bayes-notebook.ipynb)
#
# Companion notebook for the *Naive Bayes* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn class that does the same job.
#
# **Data:** the Titanic passenger list for counting categories, and the Wine recognition data
# for numeric features.

# %%
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import OrdinalEncoder

RANDOM_STATE = 42
ALPHA = 1

# %% [markdown]
# ## 1. Bayes' rule
#
# $P(c \mid \mathbf{x}) = \dfrac{P(\mathbf{x} \mid c)\,P(c)}{P(\mathbf{x})}$. With the numbers from the slides:

# %%
prior_spam, likelihood_spam, likelihood_ham = 0.3, 0.8, 0.0857
evidence = likelihood_spam * prior_spam + likelihood_ham * (1 - prior_spam)
print(f"P(spam | word) = {likelihood_spam * prior_spam / evidence:.2f}")

# %% [markdown]
# ## 2. Counting with smoothing
#
# Naive Bayes multiplies one probability per feature: $\hat{y} = \arg\max_c P(c)\prod_j P(x_j \mid c)$.
# Each probability is a count, with Laplace smoothing so no count is zero:
# $P(x_j = v \mid c) = \dfrac{N_{jvc} + \alpha}{N_c + \alpha V_j}$, where $V_j$ is the number of values of feature $j$.

# %%
titanic = sns.load_dataset("titanic")[["sex", "pclass", "embarked", "survived"]].dropna()
X = OrdinalEncoder().fit_transform(titanic[["sex", "pclass", "embarked"]]).astype(int)
y = titanic["survived"].to_numpy()

classes = np.unique(y)
priors = np.array([np.mean(y == c) for c in classes])
scores = np.tile(np.log(priors), (len(X), 1))
for j in range(X.shape[1]):
    value_count = X[:, j].max() + 1
    for c in classes:
        counts = np.bincount(X[y == c, j], minlength=value_count)
        likelihood = (counts + ALPHA) / (np.sum(y == c) + ALPHA * value_count)
        scores[:, c] += np.log(likelihood[X[:, j]])

probabilities_by_hand = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

# %%
model = CategoricalNB(alpha=ALPHA).fit(X, y)
print("same probabilities:", np.allclose(probabilities_by_hand, model.predict_proba(X)))

# %% [markdown]
# Adding logs instead of multiplying probabilities gives the same winner and avoids numbers that are too small for the computer.
#
# ## 3. Numeric features: the bell curve
#
# For a number, $P(x_j \mid c)$ is a normal curve with the class mean $\mu_{jc}$ and variance $\sigma_{jc}^2$.

# %%
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,
                                                    random_state=RANDOM_STATE, stratify=wine.target)

model = GaussianNB().fit(X_train, y_train)
extra_variance = model.epsilon_            # a tiny amount scikit-learn adds to every variance

log_scores = []
for c in np.unique(y_train):
    rows = X_train[y_train == c]
    mean, variance = rows.mean(axis=0), rows.var(axis=0) + extra_variance
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + (X_test - mean) ** 2 / variance, axis=1)
    log_scores.append(np.log(len(rows) / len(X_train)) + log_likelihood)
predictions_by_hand = np.argmax(log_scores, axis=0)

# %%
print("same predictions:", np.array_equal(predictions_by_hand, model.predict(X_test)))
print(f"test accuracy {model.score(X_test, y_test):.3f}")

# %% [markdown]
# ## Exercises
#
# 1. Set `ALPHA = 0` in section 2. Which probabilities become 0, and why is that a problem?
# 2. Add `alone` as a fourth Titanic feature and check the result against `CategoricalNB` again.
# 3. Print the class means of `alcohol` in the Wine data. Which class does a wine with 14% alcohol favor?
