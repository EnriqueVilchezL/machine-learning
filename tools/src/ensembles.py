# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ensemble methods
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/03-supervised/ensembles/ensembles-notebook.ipynb)
#
# Companion notebook for the *Ensemble methods* slides. Each idea is written first by hand
# with NumPy and then with the scikit-learn class that does the same job.
#
# **Data:** the Breast Cancer Wisconsin data (569 tumors, 30 measurements, malignant or benign).

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble, tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TREE_COUNT = 100
ROUNDS = 5
MIN_AGREEMENT = 0.95

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
rng = np.random.default_rng(RANDOM_STATE)
n = len(y_train)

# %% [markdown]
# ## 1. A bootstrap sample
#
# Draw $n$ rows with replacement. Each row is left out with probability $(1 - 1/n)^n \approx 0.368$.

# %%
sample = rng.integers(0, n, size=n)
left_out = 1 - len(np.unique(sample)) / n
left_out_formula = (1 - 1 / n) ** n
print(f"left out: {left_out:.3f} | formula: {left_out_formula:.3f}")

# %% [markdown]
# ## 2. Bagging
#
# Train one tree on each of $B$ bootstrap samples and let them vote: $\hat{y} = \arg\max_c \sum_{b=1}^{B} \mathbb{I}(\hat{f}_b(\mathbf{x}) = c)$.
# The base trees come from scikit-learn; the bagging is ours.

# %%
class BaggingClassifier:
    """Majority vote of decision trees, each trained on its own bootstrap sample."""

    def __init__(self, n_estimators: int = 10, random_state: int | None = None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            sample = rng.integers(0, len(X), size=len(X))             # bootstrap sample
            estimator = tree.DecisionTreeClassifier().fit(X[sample], y[sample])
            self.estimators_.append(estimator)
        return self

    def predict(self, X):
        vote_counts = np.zeros((len(self.classes_), len(X)), dtype=int)
        for estimator in self.estimators_:
            prediction = estimator.predict(X)                          # f_b(x)
            for i, c in enumerate(self.classes_):
                vote_counts[i] += prediction == c                      # I(f_b(x) = c)
        winner = np.argmax(vote_counts, axis=0)                        # argmax over c
        return self.classes_[winner]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


by_hand = BaggingClassifier(n_estimators=TREE_COUNT, random_state=RANDOM_STATE).fit(X_train, y_train)

# %% [markdown]
# The two versions draw different random samples, so their trees differ. Instead of equal predictions we ask that
# they agree on at least 95% (`MIN_AGREEMENT`) of the test tumors.

# %%
single_tree = tree.DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
bagging = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators=TREE_COUNT,
                                     random_state=RANDOM_STATE).fit(X_train, y_train)

agreement = np.mean(by_hand.predict(X_test) == bagging.predict(X_test))
accuracy_tree = single_tree.score(X_test, y_test)
accuracy_by_hand = by_hand.score(X_test, y_test)
accuracy_library = bagging.score(X_test, y_test)
print(f"test accuracy: one tree {accuracy_tree:.3f}, "
      f"bagging by hand {accuracy_by_hand:.3f}, scikit-learn {accuracy_library:.3f}")
print(f"agreement {agreement:.3f} | at least {MIN_AGREEMENT}:", agreement >= MIN_AGREEMENT)

# %% [markdown]
# ## 3. Random forests
#
# A random forest is bagging plus one more twist: each split only looks at a random subset of
# about $\sqrt{d}$ features, so the trees differ more.

# %%
forest = ensemble.RandomForestClassifier(n_estimators=TREE_COUNT, max_features="sqrt",
                                         random_state=RANDOM_STATE).fit(X_train, y_train)
d = X.shape[1]
features_per_split = int(np.sqrt(d))
forest_accuracy = forest.score(X_test, y_test)
print(f"features per split: {features_per_split} of {d} | test accuracy {forest_accuracy:.3f}")

# %% [markdown]
# ## 4. Boosting
#
# AdaBoost trains one-split trees $h_m$ one after another, with labels $y_i \in \{-1, +1\}$.
# A tree with weighted error $\varepsilon_m$ gets the vote $\alpha_m = \frac{1}{2}\ln\frac{1 - \varepsilon_m}{\varepsilon_m}$,
# and the weights become $w_i \leftarrow w_i\, e^{-\alpha_m y_i h_m(\mathbf{x}_i)}$, rescaled to add up to 1:
# the examples it got wrong weigh more in the next round. The prediction is $\operatorname{sign}\big(\sum_m \alpha_m h_m(\mathbf{x})\big)$.

# %%
class AdaBoostClassifier:
    """AdaBoost for two classes: a weighted vote of one-split trees."""

    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        signs = np.where(y == self.classes_[1], 1, -1)                 # y_i in {-1, +1}
        weights = np.full(len(X), 1 / len(X))                         # w_i
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        for _ in range(self.n_estimators):
            stump = tree.DecisionTreeClassifier(max_depth=1).fit(X, signs, sample_weight=weights)
            h = stump.predict(X)                                       # h_m(x_i)
            error = np.sum(weights[h != signs])                        # epsilon_m
            alpha = 0.5 * np.log((1 - error) / error)                  # alpha_m
            weights = weights * np.exp(-alpha * signs * h)
            weights /= weights.sum()                                   # rescale to add up to 1
            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)
        return self

    def decision_function(self, X):
        total = 0
        for alpha, stump in zip(self.estimator_weights_, self.estimators_):
            total = total + alpha * stump.predict(X)                   # sum_m alpha_m h_m(x)
        return total

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


by_hand = AdaBoostClassifier(n_estimators=ROUNDS).fit(X_train, y_train)
rounded_errors = np.round(by_hand.estimator_errors_, 3)
rounded_votes = np.round(by_hand.estimator_weights_, 3)
print("errors:", rounded_errors, "| votes:", rounded_votes)

# %% [markdown]
# scikit-learn uses the vote $\ln\frac{1 - \varepsilon_m}{\varepsilon_m}$, twice ours. Doubling every vote does not change
# any sign, so the predictions are the same.

# %%
boosting = ensemble.AdaBoostClassifier(n_estimators=ROUNDS, random_state=RANDOM_STATE).fit(X_train, y_train)
print("same errors:", np.allclose(by_hand.estimator_errors_, boosting.estimator_errors_))
print("votes are half:", np.allclose(by_hand.estimator_weights_, boosting.estimator_weights_ / 2))
print("same predictions:", np.array_equal(by_hand.predict(X_test), boosting.predict(X_test)))

# %% [markdown]
# ## 5. Comparing them

# %%
boosting = ensemble.AdaBoostClassifier(n_estimators=TREE_COUNT, random_state=RANDOM_STATE).fit(X_train, y_train)
names = ["one tree", "bagging", "random forest", "AdaBoost"]
accuracies = [model.score(X_test, y_test) for model in [single_tree, bagging, forest, boosting]]
plt.bar(names, accuracies)
plt.ylim(0.85, 1)
plt.ylabel("test accuracy")
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. Repeat section 2 with 1, 5, 25 and 100 trees. How does the accuracy change?
# 2. Set `max_features=None` in the forest. It is now plain bagging; compare the accuracy.
# 3. Train the AdaBoost of section 4 with 1, 5, 25 and 100 rounds. Does the training accuracy keep rising? And the test accuracy?
