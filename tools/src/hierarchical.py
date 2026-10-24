# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hierarchical clustering
#
# Companion notebook for the *Hierarchical clustering* slides. Each idea is written first by hand
# with NumPy and then with SciPy or scikit-learn doing the same job.
#
# **Data:** 30 synthetic points in 2-D from `make_blobs`, generated around 3 centers. The labels are thrown away.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

RANDOM_STATE = 42
K = 3

X, _ = make_blobs(n_samples=30, centers=K, cluster_std=1.5, random_state=RANDOM_STATE)

# %% [markdown]
# ## 1. Distance between examples
#
# $d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_j (a_j - b_j)^2}$, for every pair of examples.

# %%
distances_by_hand = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
print("distance table:", distances_by_hand.shape)
print("same result:", np.allclose(distances_by_hand, squareform(pdist(X))))

# %% [markdown]
# ## 2. Merging the closest pair
#
# Start with every example alone and merge $(A^*, B^*) = \arg\min_{A \neq B} D(A, B)$, $n - 1$ times.
# The linkage $D$ is the smallest (single), largest (complete) or mean (average) distance over all pairs $\mathbf{a} \in A$, $\mathbf{b} \in B$.

# %%
for method, rule in {"single": np.min, "complete": np.max, "average": np.mean}.items():
    clusters = [[i] for i in range(len(X))]
    heights = []
    while len(clusters) > 1:
        height, a, b = min((rule(distances_by_hand[np.ix_(A, B)]), a, b)
                           for a, A in enumerate(clusters) for b, B in enumerate(clusters) if a < b)
        heights.append(height)
        clusters[a] += clusters.pop(b)
    print(f"{method:8} {len(heights)} merges | same heights:", np.allclose(np.sort(heights), np.sort(linkage(X, method)[:, 2])))

# %% [markdown]
# ## 3. Ward's linkage
#
# Ward merges the pair whose within-cluster sum of squares $W(A) = \sum_{\mathbf{a} \in A} \lVert \mathbf{a} - \boldsymbol{\mu}_A \rVert^2$ grows the least:
# $\Delta(A, B) = \frac{|A|\,|B|}{|A| + |B|} \lVert \boldsymbol{\mu}_A - \boldsymbol{\mu}_B \rVert^2$.
# We check it on the last merge, which joins the two clusters of the cut into $K = 2$.
# SciPy reports the height $\sqrt{2\Delta}$.

# %%
ward_tree = linkage(X, "ward")
two = fcluster(ward_tree, t=2, criterion="maxclust")
A, B = X[two == 1], X[two == 2]


def within(C: np.ndarray) -> float:
    """Sum of squared distances from each example to the mean of its cluster."""
    return ((C - C.mean(axis=0)) ** 2).sum()


delta = len(A) * len(B) / (len(A) + len(B)) * ((A.mean(axis=0) - B.mean(axis=0)) ** 2).sum()
print("formula matches W growth:", np.isclose(delta, within(np.vstack([A, B])) - within(A) - within(B)))
print("same height:", np.isclose(np.sqrt(2 * delta), ward_tree[-1, 2]))

# %% [markdown]
# ## 4. Reading and cutting the dendrogram
#
# Each join sits at the height of its merge. A cut across the tree gives one cluster per branch it crosses;
# a good cut sits in a long gap between joins.

# %%
average_tree = linkage(X, "average")
dendrogram(average_tree)
plt.ylabel("merge distance")
plt.show()

# %%
cut_labels = fcluster(average_tree, t=K, criterion="maxclust")
model = AgglomerativeClustering(n_clusters=K, linkage="average").fit(X)
print("same grouping:", adjusted_rand_score(cut_labels, model.labels_) == 1.0)

# %% [markdown]
# ## Exercises
#
# 1. Add the point `(30, 30)` to `X` and draw the dendrogram again. Where does it join?
# 2. Cut the average-linkage tree at the height `t=5` with `criterion="distance"`. How many clusters do you get?
# 3. Multiply the second feature by 100 and compare the single-linkage clusters before and after, with `adjusted_rand_score`.
