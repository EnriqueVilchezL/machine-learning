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
# with NumPy and then checked against the exact derivative and against finite differences from SciPy.
#
# **Data:** no data set. The running example $f(x_1, x_2) = \ln x_1 + x_1 x_2 - \sin x_2$ at $(2, 5)$
# and the one-neuron loss of the slides.

# %%
import numpy as np
from scipy.optimize import approx_fprime

RANDOM_STATE = 42
STEP_H = 1e-6
x1, x2 = 2.0, 5.0


def f(x: np.ndarray) -> float:
    """The running example."""
    return np.log(x[0]) + x[0] * x[1] - np.sin(x[1])


exact = np.array([1 / x1 + x2, x1 - np.cos(x2)])  # derivative by formula: (1/x1 + x2, x1 - cos x2)

# %% [markdown]
# ## 1. The evaluation trace
#
# Running $f$ computes one elementary operation at a time, $v_i = \varphi_i(\text{parents of } v_i)$.

# %%
v1 = np.log(x1)
v2 = x1 * x2
v3 = np.sin(x2)
v4 = v1 + v2
v5 = v4 - v3
print(f"v1 = {v1:.3f}, v2 = {v2:.0f}, v3 = {v3:.3f}, v4 = {v4:.3f}, y = v5 = {v5:.3f}")
print("same result:", np.isclose(v5, f(np.array([x1, x2]))))

# %% [markdown]
# ## 2. Finite differences
#
# Nudge one input by a small step $h$: $\dfrac{\partial f}{\partial x_k} \approx \dfrac{f(\mathbf{x} + h\,\mathbf{e}_k) - f(\mathbf{x})}{h}$.
# The answer is close to the exact derivative, but not equal.

# %%
point = np.array([x1, x2])
by_hand = np.array([(f(point + STEP_H * e) - f(point)) / STEP_H for e in np.eye(2)])
numeric = approx_fprime(point, f, STEP_H)
print("finite differences:", by_hand.round(6), "| same as SciPy:", np.allclose(by_hand, numeric))
print("error against the exact derivative:", np.abs(by_hand - exact))

# %% [markdown]
# ## 3. Forward mode
#
# Fix one input and go from left to right: $\dfrac{\partial v_i}{\partial x_1} = \sum_{j \in \text{parents}(i)} \dfrac{\partial v_i}{\partial v_j} \dfrac{\partial v_j}{\partial x_1}$.
# One pass gives the slope for that input only, so two inputs need two passes.

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

# %%
dy_dv5 = 1.0
dy_dv4 = dy_dv5 * 1
dy_dv3 = dy_dv5 * -1
dy_dv2 = dy_dv4 * 1
dy_dv1 = dy_dv4 * 1
dy_dx1 = dy_dv1 * (1 / x1) + dy_dv2 * x2
dy_dx2 = dy_dv2 * x1 + dy_dv3 * np.cos(x2)

reverse_slopes = np.array([dy_dx1, dy_dx2])
print("reverse mode:", reverse_slopes.round(3))
print("same as exact:", np.allclose(reverse_slopes, exact), "| close to finite differences:", np.allclose(reverse_slopes, numeric, atol=1e-4))

# %% [markdown]
# ## 5. A tiny reverse-mode engine
#
# Each operation creates a node that stores its value and, for each parent, the local derivative.
# `backward` sorts the nodes so that each comes after its parents, sets the output's gradient to 1,
# and visits them from last to first, adding gradient × local derivative to each parent.

# %%
def as_value(x):
    """Wrap a plain number so it can enter the graph."""
    return x if isinstance(x, Value) else Value(x)


class Value:
    """A number that remembers how it was made."""

    def __init__(self, value: float, parents: tuple = ()):
        self.value, self.grad, self.parents = value, 0.0, parents  # parents: (node, local derivative)

    def __add__(self, other):
        other = as_value(other)
        return Value(self.value + other.value, ((self, 1.0), (other, 1.0)))

    def __sub__(self, other):
        other = as_value(other)
        return Value(self.value - other.value, ((self, 1.0), (other, -1.0)))

    def __mul__(self, other):
        other = as_value(other)
        return Value(self.value * other.value, ((self, other.value), (other, self.value)))

    def log(self):
        return Value(np.log(self.value), ((self, 1 / self.value),))

    def sin(self):
        return Value(np.sin(self.value), ((self, np.cos(self.value)),))

    def tanh(self):
        t = np.tanh(self.value)
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
# The engine on the running example gives the same slopes as reverse mode by hand.

# %%
a, b = Value(x1), Value(x2)
y = a.log() + a * b - b.sin()
y.backward()
print("engine:", np.round([a.grad, b.grad], 3), "| same as exact:", np.allclose([a.grad, b.grad], exact))

# %% [markdown]
# ## 6. Backpropagation is reverse mode
#
# One neuron, $L = (\tanh(w x + b) - y)^2$ with $x = 1.5$, $b = 0.2$, $y = 0.5$, $w = 0.5$.
# The residual $r$ is used twice in $r \cdot r$, so its two branches add up inside the engine.

# %%
def neuron_loss(w: float) -> float:
    """The loss of the neuron as a plain function of w."""
    return (np.tanh(w * 1.5 + 0.2) - 0.5) ** 2


w = Value(0.5)
residual = (w * 1.5 + 0.2).tanh() - 0.5
loss = residual * residual
loss.backward()

finite_difference = (neuron_loss(0.5 + STEP_H) - neuron_loss(0.5 - STEP_H)) / (2 * STEP_H)
print(f"L = {loss.value:.4f}, engine dL/dw = {w.grad:.4f}, finite difference = {finite_difference:.4f}")
print("same result:", np.isclose(w.grad, finite_difference))

# %% [markdown]
# ## Exercises
#
# 1. Add a `__pow__` method for $u^2$ (local derivative $2u$) and check that `residual ** 2` gives the same `w.grad`.
# 2. Take ten gradient steps $w \leftarrow w - 0.5 \, \partial L / \partial w$ with the engine and print the loss after each one.
# 3. In section 4, keep only the last term of $\partial y / \partial x_1$ (forget the branch through $v_1$) and compare with the exact value.
