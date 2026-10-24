# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Convolutional neural networks
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/06-deep-learning/cnn/cnn-notebook.ipynb)
#
# Companion notebook for the *Convolutional neural networks* slides. Each idea is written first by hand
# with NumPy, in small classes shaped like PyTorch's, and then with the PyTorch module that does the same job.
#
# **Data:** the Digits data (1 797 handwritten digits, each a grid of 8 × 8 gray pixels).

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

RANDOM_STATE = 42
TRAINING_STEPS = 100

torch.manual_seed(RANDOM_STATE)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(RANDOM_STATE)

digits = load_digits()
images = digits.images / 16
image = images[0][None]  # one channel: shape (C, H, W) = (1, 8, 8), the layout PyTorch uses
image_t = torch.tensor(image)
print("image shape:", image.shape)

# %% [markdown]
# ## 1. The convolution
#
# Place the $k \times k$ kernel with its corner at row $i$, column $j$, multiply, add up and add the bias:
# $Z_{ij} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X_{i+m,\,j+n} \, K_{mn} + b$.
# The NumPy `Conv2d` below has the same arguments as `nn.Conv2d` and keeps the kernels in `weight`,
# shaped (filters, channels, $k$, $k$). It already handles channels, padding and stride, used in the next sections.
# This kernel adds the left column of each patch and subtracts the right one: it finds vertical edges.

# %%
class Conv2d:
    """Slide each kernel over one image of shape (C, H, W), like nn.Conv2d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        self.weight = rng.normal(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    def __call__(self, X: np.ndarray) -> np.ndarray:
        p = self.padding
        s = self.stride
        k = self.weight.shape[-1]
        X = np.pad(X, ((0, 0), (p, p), (p, p)))  # border of p zeros
        H_out = (X.shape[1] - k) // s + 1
        W_out = (X.shape[2] - k) // s + 1
        Z = np.zeros((len(self.weight), H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                top, left = i * s, j * s                       # corner of the window
                window = X[:, top:top + k, left:left + k]      # X_{i+m, j+n} for all m, n
                products = self.weight * window                # X_{i+m, j+n} K_{mn}, each filter
                total = products.sum(axis=(1, 2, 3))           # sum over c, m, n
                Z[:, i, j] = total + self.bias                 # + b
        return Z


def copy_weights(mine: Conv2d, theirs: nn.Conv2d) -> None:
    """Put the NumPy kernels and biases into the torch layer."""
    with torch.no_grad():
        theirs.weight.copy_(torch.from_numpy(mine.weight))
        theirs.bias.copy_(torch.from_numpy(mine.bias))


vertical_edge = np.array([[1.0, 0, -1], [1, 0, -1], [1, 0, -1]])
conv = Conv2d(1, 1, kernel_size=3)
conv.weight = vertical_edge[None, None]
feature_map = conv(image)

torch_conv = nn.Conv2d(1, 1, kernel_size=3)
copy_weights(conv, torch_conv)
print("feature map shape:", feature_map.shape)
print("same result:", np.allclose(feature_map, torch_conv(image_t).detach().numpy()))

# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
axes[0].imshow(image[0], cmap="gray_r")
axes[0].set_title("image $\\mathbf{X}$")
limit = np.abs(feature_map).max()
axes[1].imshow(feature_map[0], cmap="RdBu_r", vmin=-limit, vmax=limit)
axes[1].set_title("feature map $\\mathbf{Z}$")
for ax in axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# ## 2. Padding and stride
#
# Padding $p$ adds a border of zeros; stride $s$ is how far the kernel jumps.
# The output side is $n_{\text{out}} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$.
# Besides the layer `nn.Conv2d`, PyTorch has the plain function `F.conv2d`, which takes the kernel as an argument.

# %%
def output_size(n: int, k: int, p: int, s: int) -> int:
    return (n + 2 * p - k) // s + 1


kernel_t = torch.tensor(vertical_edge[None, None])
for p, s in [(1, 1), (0, 2)]:
    conv = Conv2d(1, 1, kernel_size=3, stride=s, padding=p)
    conv.weight = vertical_edge[None, None]
    by_hand = conv(image)
    torch_result = F.conv2d(image_t, kernel_t, stride=s, padding=p)
    formula = output_size(8, 3, p, s)
    same = np.allclose(by_hand, torch_result.numpy())
    print(f"p = {p}, s = {s}: shape {by_hand.shape[1:]}, formula {formula} | same result:", same)

# %% [markdown]
# ## 3. Channels, filters and ReLU
#
# With $C$ channels each kernel is $k \times k \times C$ and sums over all of them:
# $Z^{(f)}_{ij} = \sum_{c} \sum_{m} \sum_{n} X_{i+m,\,j+n,\,c} \, K^{(f)}_{mnc} + b^{(f)}$.
# ReLU then keeps the positive entries: $A_{ij} = \max(0, Z_{ij})$.
# Here three digits play the role of the three color channels.

# %%
color_image = images[:3]  # shape (C, H, W) = (3, 8, 8)
conv = Conv2d(3, 1, kernel_size=3)
conv.bias = np.array([0.5])
Z = conv(color_image)
A = np.maximum(0, Z)  # ReLU

torch_layer = nn.Sequential(nn.Conv2d(3, 1, kernel_size=3), nn.ReLU())
copy_weights(conv, torch_layer[0])
A_torch = torch_layer(torch.tensor(color_image))
print("same result:", np.allclose(A, A_torch.detach().numpy()))
print(f"ReLU sets {np.mean(A == 0):.0%} of the entries to 0")
count_by_hand = conv.weight.size + conv.bias.size
count_torch = sum(p.numel() for p in torch_layer.parameters())
print("parameters of one filter, k*k*C + 1 =", count_by_hand, "| torch counts", count_torch)

# %% [markdown]
# ## 4. Pooling
#
# Max pooling keeps the largest value of each $2 \times 2$ window (stride 2), so the map shrinks by half.
# It has no weights to learn.

# %%
class MaxPool2d:
    """Largest value of each k x k window, stride k, like nn.MaxPool2d."""

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def __call__(self, A: np.ndarray) -> np.ndarray:
        k = self.kernel_size
        C, H, W = A.shape
        H_out = H // k
        W_out = W // k
        pooled = np.zeros((C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                window = A[:, i * k:(i + 1) * k, j * k:(j + 1) * k]
                pooled[:, i, j] = window.max(axis=(1, 2))  # largest value, each channel
        return pooled


pooled_by_hand = MaxPool2d(2)(A)
pooled_torch = nn.MaxPool2d(2)(torch.tensor(A))
same = np.allclose(pooled_by_hand, pooled_torch.numpy())
print("shape", A.shape, "->", pooled_by_hand.shape, "| same result:", same)

# %% [markdown]
# ## 5. The gradient of a shared weight
#
# Each kernel weight is used at every position, so its gradient adds up all of them:
# $\dfrac{\partial L}{\partial K_{mn}} = \sum_i \sum_j \dfrac{\partial L}{\partial Z_{ij}} \, X_{i+m,\,j+n}$.
# With the toy loss $L = \frac{1}{2}\sum_{ij} Z_{ij}^2$ we have $\partial L / \partial Z_{ij} = Z_{ij}$.
# PyTorch's autograd finds the same gradient in `weight.grad` after `loss.backward()`.

# %%
conv = Conv2d(1, 1, kernel_size=3)
dL_dZ = conv(image)[0]  # dL/dZ_ij = Z_ij
Z_side = dL_dZ.shape[0]
gradient_by_hand = np.zeros((3, 3))
for m in range(3):
    for n in range(3):
        window = image[0, m:m + Z_side, n:n + Z_side]       # X_{i+m, j+n} for all i, j
        gradient_by_hand[m, n] = np.sum(dL_dZ * window)     # sum over i, j

torch_conv = nn.Conv2d(1, 1, kernel_size=3)
copy_weights(conv, torch_conv)
Z_t = torch_conv(image_t)
loss = 0.5 * (Z_t ** 2).sum()
loss.backward()
print("same result:", np.allclose(gradient_by_hand, torch_conv.weight.grad[0, 0].numpy()))

# %% [markdown]
# ## 6. A small CNN
#
# Convolution, ReLU and pooling, then a dense layer to the 10 digits, trained with `nn.CrossEntropyLoss` and `torch.optim.Adam`.
# PyTorch layers work on a batch, so the images enter with shape (examples, channels, $H$, $W$).
# Eight filters of size $3 \times 3$ with $p = 1$ keep the $8 \times 8$ size, and pooling halves it.

# %%
X_train, X_test, y_train, y_test = train_test_split(images[:, None], digits.target, test_size=0.3,
                                                    random_state=RANDOM_STATE, stratify=digits.target)
X_train_t = torch.tensor(X_train)
X_test_t = torch.tensor(X_test)
y_train_t = torch.tensor(y_train)

side = output_size(8, 3, 1, 1) // 2
cnn = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(8 * side * side, 10))
before_dense = cnn[:3](X_test_t)  # after convolution, ReLU and pooling
print("shape before the dense layer:", tuple(before_dense.shape[1:]), "| formula:", (8, side, side))

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
cross_entropy = nn.CrossEntropyLoss()
for _ in range(TRAINING_STEPS):
    optimizer.zero_grad()
    prediction = cnn(X_train_t)
    loss = cross_entropy(prediction, y_train_t)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = cnn(X_test_t).argmax(dim=1).numpy()
print(f"training cost {loss.item():.3f}, test accuracy {np.mean(predictions == y_test):.3f}")

# %% [markdown]
# ## Exercises
#
# 1. Use the horizontal-edge kernel (the transpose of `vertical_edge`) in section 1 and compare both feature maps.
# 2. Write an `AvgPool2d` class like `MaxPool2d` and compare it with `nn.AvgPool2d(2)`.
# 3. Count the parameters of a layer with 64 filters of size $3 \times 3$ on a color image (check with `nn.Conv2d(3, 64, 3)`),
#    and of a dense layer from a $200 \times 200 \times 3$ image to 1 000 neurons.
