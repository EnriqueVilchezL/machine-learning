# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Recurrent neural networks
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/06-deep-learning/rnn/rnn-notebook.ipynb)
#
# Companion notebook for the *Recurrent neural networks* slides. Each idea is written first by hand
# with NumPy, in small classes shaped like PyTorch's, and then with the PyTorch module that does the same job.
#
# **Data:** a synthetic sine wave with a little noise, cut into windows of $T$ steps; the target is the next value.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from torch import nn

RANDOM_STATE = 42
T = 10
HIDDEN = 8
LEARNING_RATE = 0.05
STEPS = 2000
CLIP = 1.0

torch.manual_seed(RANDOM_STATE)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(RANDOM_STATE)
series = np.sin(np.arange(400) * 0.2) + 0.05 * rng.normal(size=400)
windows = np.lib.stride_tricks.sliding_window_view(series, T + 1).copy()
X, y = windows[:, :T, None], windows[:, T:]
X_train, y_train = X[:300], y[:300]
X_test, y_test = X[300:], y[300:]

# %% [markdown]
# ## 1. The recurrent cell
#
# $\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$, starting at $\mathbf{h}_0 = \mathbf{0}$.
# The class uses PyTorch's names: `weight_ih` is $\mathbf{W}_{xh}$, `weight_hh` is $\mathbf{W}_{hh}$, and
# PyTorch splits $\mathbf{b}_h$ into two biases, $\mathbf{b}_h$ = `bias_ih` + `bias_hh`.

# %%
class RNN:
    """NumPy version of nn.RNN (one layer, tanh, batch_first=True)."""

    def __init__(self, input_size: int, hidden_size: int):
        self.weight_ih = rng.normal(0, 0.5, (hidden_size, input_size))
        self.weight_hh = rng.normal(0, 0.3, (hidden_size, hidden_size))
        self.bias_ih = np.zeros(hidden_size)
        self.bias_hh = np.zeros(hidden_size)

    def forward(self, x: np.ndarray, h0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x has shape (batch, T, input_size); returns every state h_1..h_T and the last one."""
        h = h0
        states = []
        for t in range(x.shape[1]):
            x_t = x[:, t]
            input_term = x_t @ self.weight_ih.T + self.bias_ih      # W_xh x_t
            recurrent_term = h @ self.weight_hh.T + self.bias_hh    # W_hh h_{t-1}
            h = np.tanh(input_term + recurrent_term)                # h_t
            states.append(h)
        outputs = np.stack(states, axis=1)
        return outputs, outputs[:, -1]


def copy_weights(source: RNN, target: nn.RNN) -> None:
    """Copy the NumPy weights into the PyTorch module (its names end in _l0, for layer 0)."""
    with torch.no_grad():
        for name in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
            getattr(target, name + "_l0").copy_(torch.from_numpy(getattr(source, name)))

# %% [markdown]
# The slide's example: one state number, $W_{xh} = 1$, $W_{hh} = 0.5$, inputs $1, 0, 1$.

# %%
cell = RNN(1, 1)
cell.weight_ih = np.array([[1.0]])  # W_xh
cell.weight_hh = np.array([[0.5]])  # W_hh
inputs = np.array([[[1.0], [0.0], [1.0]]])
outputs, h_n = cell.forward(inputs, np.zeros((1, 1)))
print("states:", outputs.ravel().round(3))

torch_cell = nn.RNN(1, 1, batch_first=True)
copy_weights(cell, torch_cell)
torch_outputs, torch_h_n = torch_cell(torch.from_numpy(inputs), torch.zeros(1, 1, 1))
print("same result:", np.allclose(outputs, torch_outputs.detach().numpy()))

# %% [markdown]
# ## 2. Backpropagation through time
#
# The network reads a window and predicts the next value from the last state, $\hat{y} = \mathbf{W}_{hy}\mathbf{h}_T + b_y$,
# with squared error as the loss. Backpropagation runs back along the states, and the shared weights add up the gradients of all steps.

# %%
def bptt(rnn: RNN, x: np.ndarray, outputs: np.ndarray, d_h: np.ndarray) -> list[np.ndarray]:
    """Gradients of weight_ih, weight_hh, bias_ih, bias_hh, given d_h = dJ/dh_T.

    Walks back along the states h_T, ..., h_1 (with h_0 = 0)."""
    states = np.concatenate([np.zeros_like(outputs[:, :1]), outputs], axis=1)  # h_0, h_1, ..., h_T
    grad_ih = 0
    grad_hh = 0
    grad_b = 0
    for t in range(x.shape[1], 0, -1):
        h_t = states[:, t]
        h_prev = states[:, t - 1]
        x_t = x[:, t - 1]
        d_a = d_h * (1 - h_t ** 2)       # back through tanh: tanh' = 1 - h_t^2
        grad_ih = grad_ih + d_a.T @ x_t      # this step's share of dJ/dW_xh
        grad_hh = grad_hh + d_a.T @ h_prev   # this step's share of dJ/dW_hh
        grad_b = grad_b + d_a.sum(axis=0)
        d_h = d_a @ rnn.weight_hh        # dJ/dh_{t-1}
    return [grad_ih, grad_hh, grad_b, grad_b]


rnn = RNN(1, HIDDEN)
head_weight = rng.normal(0, 0.5, (1, HIDDEN))  # W_hy
head_bias = np.zeros(1)                        # b_y


def gradients(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Forward pass, then the gradients of every weight.

    Returns the RNN's four and the head's two (W_hy, b_y)."""
    outputs, h_n = rnn.forward(X, np.zeros((len(X), HIDDEN)))
    y_hat = h_n @ head_weight.T + head_bias   # W_hy h_T + b_y
    d_y = 2 * (y_hat - y) / len(y)            # dJ/dy_hat for the mean squared error
    d_h = d_y @ head_weight                   # dJ/dh_T
    rnn_grads = bptt(rnn, X, outputs, d_h)
    grad_head_weight = d_y.T @ h_n
    grad_head_bias = d_y.sum(axis=0)
    return rnn_grads + [grad_head_weight, grad_head_bias]

# %% [markdown]
# PyTorch's autograd finds the same gradients from the forward pass alone: `loss.backward()` fills every `.grad`.

# %%
torch_rnn = nn.RNN(1, HIDDEN, batch_first=True)
torch_head = nn.Linear(HIDDEN, 1)
copy_weights(rnn, torch_rnn)
with torch.no_grad():
    torch_head.weight.copy_(torch.from_numpy(head_weight))
    torch_head.bias.copy_(torch.from_numpy(head_bias))
torch_params = list(torch_rnn.parameters()) + list(torch_head.parameters())


def torch_predict(X: np.ndarray) -> torch.Tensor:
    _, h_n = torch_rnn(torch.from_numpy(X))  # h_0 defaults to zeros
    return torch_head(h_n[-1])


loss_fn = nn.MSELoss()
y_train_t = torch.from_numpy(y_train)
prediction = torch_predict(X_train)
loss = loss_fn(prediction, y_train_t)
loss.backward()

grads_by_hand = gradients(X_train, y_train)
same = all(np.allclose(g, p.grad.numpy()) for g, p in zip(grads_by_hand, torch_params))
print("same result:", same)

# %% [markdown]
# ## 3. Vanishing and exploding gradients
#
# With one state number, $\dfrac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} w_{hh}\,(1 - h_t^2)$: one factor per step.

# %%
def run_scalar(h_1: float, w_hh: float, inputs: np.ndarray) -> np.ndarray:
    states = [h_1]
    for x_t in inputs:
        states.append(np.tanh(w_hh * states[-1] + x_t))
    return np.array(states)


inputs = rng.normal(0, 0.5, 20)
states = run_scalar(0.1, 0.8, inputs)
factors = 0.8 * (1 - states[1:] ** 2)  # w_hh (1 - h_t^2), for t = 2..T
product = np.prod(factors)

# %% [markdown]
# Autograd follows the same chain of steps backwards and gives the same number.

# %%
h_1 = torch.tensor(0.1, requires_grad=True)
h = h_1
for x_t in torch.from_numpy(inputs):
    h = torch.tanh(0.8 * h + x_t)
(autograd,) = torch.autograd.grad(h, h_1)
print(f"dh_T/dh_1 = {product:.2e} | same result:", np.isclose(product, autograd.item()))

# %%
for w_hh in [0.5, 0.8, 1.5]:
    states = run_scalar(0.0, w_hh, inputs)
    factors = w_hh * (1 - states[1:] ** 2)
    gradient_sizes = np.abs(np.cumprod(factors[::-1]))  # product of the last k factors
    plt.semilogy(gradient_sizes, marker="o", label=f"$w_{{hh}}$ = {w_hh}")
plt.xlabel("steps back")
plt.ylabel("gradient size")
plt.legend()
plt.show()

# %% [markdown]
# ## 4. Training and forecasting
#
# Gradient descent with the BPTT gradient, $\boldsymbol\theta \leftarrow \boldsymbol\theta - \alpha \nabla J$.
# Clipping scales a gradient longer than a threshold back to that length. The NumPy optimizer copies the shape of
# `torch.optim.SGD`; PyTorch's `clip_grad_norm_` adds $10^{-6}$ to the length, so we do the same.

# %%
class SGD:
    """NumPy version of torch.optim.SGD.

    step() takes the gradients, because NumPy arrays have no .grad."""

    def __init__(self, params: list[np.ndarray], lr: float):
        self.params = params
        self.lr = lr

    def step(self, grads: list[np.ndarray]) -> None:
        for param, grad in zip(self.params, grads):
            param -= self.lr * grad


params = [rnn.weight_ih, rnn.weight_hh, rnn.bias_ih, rnn.bias_hh, head_weight, head_bias]
optimizer = SGD(params, lr=LEARNING_RATE)
for _ in range(STEPS):
    grads = gradients(X_train, y_train)
    norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))  # length of all gradients together
    scale = min(1, CLIP / (norm + 1e-6))                 # < 1 only if the length passes CLIP
    clipped = [g * scale for g in grads]
    optimizer.step(clipped)

# %% [markdown]
# The PyTorch loop, from the same starting weights: `zero_grad`, `backward`, `clip_grad_norm_`, `step`.

# %%
torch_optimizer = torch.optim.SGD(torch_params, lr=LEARNING_RATE)
for _ in range(STEPS):
    torch_optimizer.zero_grad()
    prediction = torch_predict(X_train)
    loss = loss_fn(prediction, y_train_t)
    loss.backward()
    nn.utils.clip_grad_norm_(torch_params, CLIP)
    torch_optimizer.step()

same = all(np.allclose(p, q.detach().numpy()) for p, q in zip(optimizer.params, torch_params))
print("same weights:", same)

# %% [markdown]
# A Ridge regression on the same $T$ past values (lag features) is a reference forecaster.

# %%
rnn_mse = mean_squared_error(y_test, torch_predict(X_test).detach().numpy())
ridge = Ridge(alpha=1.0).fit(X_train[:, :, 0], y_train)
ridge_mse = mean_squared_error(y_test, ridge.predict(X_test[:, :, 0]))
last_value_mse = mean_squared_error(y_test, X_test[:, -1])
print(f"test MSE: RNN {rnn_mse:.4f}, Ridge {ridge_mse:.4f}, repeat last value {last_value_mse:.4f}")
both_beat = rnn_mse < last_value_mse and ridge_mse < last_value_mse
print("both beat repeating the last value:", both_beat)

# %% [markdown]
# ## 5. The LSTM cell
#
# Gates $\mathbf{f}_t, \mathbf{i}_t, \mathbf{o}_t = \sigma(\mathbf{W}[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b})$ and candidate $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$;
# then $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ and $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$.
#
# Libraries stack the four matrices into one and do a single product per step, split into the part for $\mathbf{x}_t$ (`weight_ih`)
# and the part for $\mathbf{h}_{t-1}$ (`weight_hh`). PyTorch's order of the four blocks is $\mathbf{i}, \mathbf{f}, \tilde{\mathbf{c}}, \mathbf{o}$ (it calls $\tilde{\mathbf{c}}$ "g").

# %%
def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-a))


class LSTMCell:
    """NumPy version of nn.LSTMCell, gate blocks in PyTorch's order i, f, g, o."""

    def __init__(self, input_size: int, hidden_size: int):
        self.weight_ih = rng.normal(0, 0.5, (4 * hidden_size, input_size))
        self.weight_hh = rng.normal(0, 0.5, (4 * hidden_size, hidden_size))
        self.bias_ih, self.bias_hh = rng.normal(0, 0.1, (2, 4 * hidden_size))

    def forward(self, x: np.ndarray,
                state: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """One step: returns the new (h, c)."""
        h_prev, c_prev = state
        input_term = x @ self.weight_ih.T + self.bias_ih            # part for x_t
        recurrent_term = h_prev @ self.weight_hh.T + self.bias_hh   # part for h_{t-1}
        a = input_term + recurrent_term
        a_i, a_f, a_g, a_o = np.split(a, 4, axis=1)
        i = sigmoid(a_i)        # input gate i_t
        f = sigmoid(a_f)        # forget gate f_t
        g = np.tanh(a_g)        # candidate c~_t
        o = sigmoid(a_o)        # output gate o_t
        c = f * c_prev + i * g  # c_t
        h = o * np.tanh(c)      # h_t
        return h, c


lstm = LSTMCell(1, HIDDEN)
x_t = np.array([[0.7]])
h_prev = rng.normal(size=(1, HIDDEN))
c_prev = rng.normal(size=(1, HIDDEN))
h, c = lstm.forward(x_t, (h_prev, c_prev))

torch_lstm = nn.LSTMCell(1, HIDDEN)
with torch.no_grad():
    for name in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
        getattr(torch_lstm, name).copy_(torch.from_numpy(getattr(lstm, name)))
torch_state = (torch.from_numpy(h_prev), torch.from_numpy(c_prev))
torch_h, torch_c = torch_lstm(torch.from_numpy(x_t), torch_state)
same_h = np.allclose(h, torch_h.detach().numpy())
same_c = np.allclose(c, torch_c.detach().numpy())
print("same result:", same_h and same_c)

# %% [markdown]
# ## Exercises
#
# 1. In section 4, replace `torch.optim.SGD` by `torch.optim.Adam(torch_params, lr=0.01)`. Does the RNN's test error improve?
# 2. In section 3, find the value of $w_{hh}$ where the gradient stops shrinking over 20 steps.
# 3. Write one GRU step, $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$, and check that $\mathbf{z}_t = \mathbf{0}$ copies the old state.
#    `nn.GRUCell` uses $\mathbf{z}_t$ the other way round: there $\mathbf{z}_t = \mathbf{1}$ copies the old state.
