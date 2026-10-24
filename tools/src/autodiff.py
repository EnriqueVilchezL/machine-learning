# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Automatic differentiation
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/06-deep-learning/autodiff/autodiff-notebook.ipynb)
#
# Companion notebook for the *Automatic differentiation* slides. Each idea is written first by hand
# with NumPy and then with PyTorch's automatic differentiation (autograd), which does the same job.
#
# **Data:** no data set. The running example $f(x_1, x_2) = \ln x_1 + x_1 x_2 - \sin x_2$ at $(2, 5)$
# and the one-neuron loss of the slides.

# %%
import numpy as np
import torch
from torch import nn

RANDOM_STATE = 42
STEP_H = 1e-6
x1, x2 = 2.0, 5.0

torch.manual_seed(RANDOM_STATE)
torch.set_default_dtype(torch.float64)


def f(x: np.ndarray) -> float:
    """The running example."""
    return np.log(x[0]) + x[0] * x[1] - np.sin(x[1])


def f_torch(x: torch.Tensor) -> torch.Tensor:
    """The running example with torch operations, so autograd can follow it."""
    return torch.log(x[0]) + x[0] * x[1] - torch.sin(x[1])


exact = np.array([1 / x1 + x2, x1 - np.cos(x2)])  # derivative by formula: (1/x1 + x2, x1 - cos x2)

# %% [markdown]
# ## 1. The evaluation trace
#
# Running $f$ computes one elementary operation at a time, $v_i = \varphi_i(\text{parents of } v_i)$.
# A tensor created with `requires_grad=True` makes PyTorch record this trace as it runs:
# each result keeps in `grad_fn` the operation that made it.

# %%
v1 = np.log(x1)
v2 = x1 * x2
v3 = np.sin(x2)
v4 = v1 + v2
v5 = v4 - v3
print(f"v1 = {v1:.3f}, v2 = {v2:.0f}, v3 = {v3:.3f}, v4 = {v4:.3f}, y = v5 = {v5:.3f}")
point_t = torch.tensor([x1, x2], requires_grad=True)
y_t = f_torch(point_t)
print("last operation recorded by torch:", y_t.grad_fn.name())
print("same result:", np.isclose(v5, y_t.item()))

# %% [markdown]
# ## 2. Finite differences
#
# Nudge one input by a small step $h$: $\dfrac{\partial f}{\partial x_k} \approx \dfrac{f(\mathbf{x} + h\,\mathbf{e}_k) - f(\mathbf{x})}{h}$.
# The answer is close to the exact derivative, but not equal. Autograd, used from section 4 on, has no such error.

# %%
point = np.array([x1, x2])
numeric = np.array([(f(point + STEP_H * e) - f(point)) / STEP_H for e in np.eye(2)])
print("finite differences:", numeric.round(6), "| close to exact:", np.allclose(numeric, exact, atol=1e-4))
print("error against the exact derivative:", np.abs(numeric - exact))

# %% [markdown]
# ## 3. Forward mode
#
# Fix one input and go from left to right: $\dfrac{\partial v_i}{\partial x_1} = \sum_{j \in \text{parents}(i)} \dfrac{\partial v_i}{\partial v_j} \dfrac{\partial v_j}{\partial x_1}$.
# One pass gives the slope for that input only, so two inputs need two passes.
# With many inputs and one output (a cost) this is slow, so PyTorch's `.backward()` uses reverse mode, the next section.

# %%
def forward_mode(dx1: float, dx2: float) -> float:
    """Carry the slope of every node with respect to the chosen input."""
    dv1 = (1 / x1) * dx1
    dv2 = x2 * dx1 + x1 * dx2
    dv3 = np.cos(x2) * dx2
    dv4 = dv1 + dv2
    return dv4 - dv3


forward_slopes = np.array([forward_mode(1, 0), forward_mode(0, 1)])
print("forward mode:", forward_slopes.round(3))
print("same as exact:", np.allclose(forward_slopes, exact), "| close to finite differences:", np.allclose(forward_slopes, numeric, atol=1e-4))

# %% [markdown]
# ## 4. Reverse mode
#
# After the forward pass, go from right to left starting at $\partial y / \partial y = 1$:
# $\dfrac{\partial y}{\partial v_j} = \sum_{i \in \text{children}(j)} \dfrac{\partial y}{\partial v_i} \dfrac{\partial v_i}{\partial v_j}$.
# One pass gives the slopes for both inputs. $x_1$ and $x_2$ each feed two operations, so their two terms add up.
# This is what `y.backward()` does in PyTorch: it walks the recorded trace backwards and leaves each slope in `.grad`.

# %%
dy_dv5 = 1.0
dy_dv4 = dy_dv5 * 1
dy_dv3 = dy_dv5 * -1
dy_dv2 = dy_dv4 * 1
dy_dv1 = dy_dv4 * 1
dy_dx1 = dy_dv1 * (1 / x1) + dy_dv2 * x2
dy_dx2 = dy_dv2 * x1 + dy_dv3 * np.cos(x2)

reverse_slopes = np.array([dy_dx1, dy_dx2])
y_t.backward()
print("reverse mode:", reverse_slopes.round(3), "| torch:", point_t.grad.numpy().round(3))
print("same as exact:", np.allclose(reverse_slopes, exact), "| same as torch:", np.allclose(reverse_slopes, point_t.grad.numpy()))

# %% [markdown]
# ## 5. A tiny reverse-mode engine
#
# Each operation creates a node that stores its value and, for each parent, the local derivative.
# The names copy PyTorch's tensors: the value is in `.data`, the slope in `.grad`, and `.backward()` fills it.
# `backward` sorts the nodes so that each comes after its parents, sets the output's gradient to 1,
# and visits them from last to first, adding gradient × local derivative to each parent.

# %%
def as_value(x):
    """Wrap a plain number so it can enter the graph."""
    return x if isinstance(x, Value) else Value(x)


class Value:
    """A number that remembers how it was made."""

    def __init__(self, data: float, parents: tuple = ()):
        self.data, self.grad, self.parents = data, 0.0, parents  # parents: (node, local derivative)

    def __add__(self, other):
        other = as_value(other)
        return Value(self.data + other.data, ((self, 1.0), (other, 1.0)))

    def __sub__(self, other):
        other = as_value(other)
        return Value(self.data - other.data, ((self, 1.0), (other, -1.0)))

    def __mul__(self, other):
        other = as_value(other)
        return Value(self.data * other.data, ((self, other.data), (other, self.data)))

    def log(self):
        return Value(np.log(self.data), ((self, 1 / self.data),))

    def sin(self):
        return Value(np.sin(self.data), ((self, np.cos(self.data)),))

    def tanh(self):
        t = np.tanh(self.data)
        return Value(t, ((self, 1 - t**2),))

    def backward(self):
        order, seen = [], set()

        def visit(node):
            if node not in seen:
                seen.add(node)
                for parent, _ in node.parents:
                    visit(parent)
                order.append(node)

        visit(self)
        for node in order:
            node.grad = 1.0 if node is self else 0.0
        for node in reversed(order):
            for parent, local in node.parents:
                parent.grad += node.grad * local


# %% [markdown]
# The engine and torch run the same lines on the running example and give the same slopes.

# %%
a, b = Value(x1), Value(x2)
y = a.log() + a * b - b.sin()
y.backward()

a_t, b_t = torch.tensor(x1, requires_grad=True), torch.tensor(x2, requires_grad=True)
y_t = torch.log(a_t) + a_t * b_t - torch.sin(b_t)
y_t.backward()
print("engine:", np.round([a.grad, b.grad], 3), "| same as torch:", np.allclose([a.grad, b.grad], [a_t.grad.item(), b_t.grad.item()]))

# %% [markdown]
# ## 6. Backpropagation is reverse mode
#
# One neuron, $L = (\tanh(w x + b) - y)^2$ with $x = 1.5$, $b = 0.2$, $y = 0.5$, $w = 0.5$.
# The residual $r$ is used twice in $r \cdot r$, so its two branches add up inside the engine.
# In PyTorch the neuron is an `nn.Linear(1, 1)` layer, the loss is `nn.MSELoss`, and `loss.backward()` is backpropagation.
# A centered finite difference $\big(L(w + h) - L(w - h)\big) / 2h$ gives one more, independent check.

# %%
def neuron_loss(w: float) -> float:
    """The loss of the neuron as a plain function of w."""
    return (np.tanh(w * 1.5 + 0.2) - 0.5) ** 2


w = Value(0.5)
residual = (w * 1.5 + 0.2).tanh() - 0.5
loss = residual * residual
loss.backward()

neuron = nn.Linear(1, 1)
with torch.no_grad():
    neuron.weight.fill_(0.5)
    neuron.bias.fill_(0.2)
torch_loss = nn.MSELoss()(torch.tanh(neuron(torch.tensor([1.5]))), torch.tensor([0.5]))
torch_loss.backward()

finite_difference = (neuron_loss(0.5 + STEP_H) - neuron_loss(0.5 - STEP_H)) / (2 * STEP_H)
print(f"L = {loss.data:.4f}, engine dL/dw = {w.grad:.4f}, torch dL/dw = {neuron.weight.grad.item():.4f}")
print("same as torch:", np.isclose(w.grad, neuron.weight.grad.item()), "| close to finite difference:", np.isclose(w.grad, finite_difference))

# %% [markdown]
# ## Exercises
#
# 1. Add a `__pow__` method for $u^2$ (local derivative $2u$) and check that `residual ** 2` gives the same `w.grad`.
# 2. Take ten gradient steps $w \leftarrow w - 0.5 \, \partial L / \partial w$ with the engine, and the same ten with
#    `torch.optim.SGD([neuron.weight], lr=0.5)` (call `zero_grad()` before each `backward()`). Print both losses after each step.
# 3. In section 4, keep only the last term of $\partial y / \partial x_1$ (forget the branch through $v_1$) and compare with the exact value.
