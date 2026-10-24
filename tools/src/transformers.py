# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Transformers
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/06-deep-learning/transformers/transformers-notebook.ipynb)
#
# Companion notebook for the *Transformers* slides. Each idea is written first by hand
# with NumPy and then with the PyTorch tools that do the same job.
#
# **Data:** two tiny made-up texts written inline. The embeddings and attention matrices are random,
# standing in for learned ones.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

RANDOM_STATE = 42
D_MODEL = 8
D_K = 4
HEADS = 2
EPOCHS = 300

torch.manual_seed(RANDOM_STATE)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(RANDOM_STATE)
sentence = "the animal did not cross the street because it was tired".split()
vocabulary = sorted(set(sentence))
token_ids = np.array([vocabulary.index(word) for word in sentence])
T = len(token_ids)

# %% [markdown]
# ## 1. Tokens and embeddings
#
# Each token number picks one row of an embedding table, its vector $\mathbf{x}_t$. The rows form $\mathbf{X}$ ($T \times d_{\text{model}}$).
# Picking a row is the same as multiplying a one-hot vector by the table. The NumPy class below has the same
# shape as `nn.Embedding`: a `weight` table and a call that looks up rows.

# %%
class Embedding:
    """NumPy twin of nn.Embedding: look up rows of a weight table."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = rng.normal(size=(num_embeddings, embedding_dim))

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        return self.weight[ids]


embedding = Embedding(len(vocabulary), D_MODEL)
X = embedding(token_ids)
one_hot = np.eye(len(vocabulary))[token_ids]
X_from_one_hot = one_hot @ embedding.weight
print("X:", X.shape, "| one-hot times table, same result:", np.allclose(X, X_from_one_hot))

# %%
torch_embedding = nn.Embedding(len(vocabulary), D_MODEL)
with torch.no_grad():
    torch_embedding.weight.copy_(torch.tensor(embedding.weight))
X_t = torch_embedding(torch.tensor(token_ids))
print("same result:", np.allclose(X, X_t.detach().numpy()))

# %% [markdown]
# ## 2. Scaled dot-product attention
#
# $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, $\mathbf{V} = \mathbf{X}\mathbf{W}_V$ and
# $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$, with softmax on each row.
# The NumPy function takes the same arguments as `F.scaled_dot_product_attention` and also returns the weights $\mathbf{A}$.

# %%
def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray,
                                 attn_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """softmax(Q K^T / sqrt(d_k) + M) V on the last two axes; returns the outputs and the weights A."""
    d_k = key.shape[-1]
    scores = query @ key.swapaxes(-1, -2)  # Q K^T
    scaled = scores / np.sqrt(d_k)  # Q K^T / sqrt(d_k)
    if attn_mask is not None:
        scaled = scaled + attn_mask  # + M
    exp_scores = np.exp(scaled - scaled.max(axis=-1, keepdims=True))  # shift for stability
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # A = softmax by rows
    output = weights @ value  # A V
    return output, weights


W_Q, W_K, W_V = (rng.normal(size=(D_MODEL, D_K)) / np.sqrt(D_MODEL) for _ in range(3))
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
output, A = scaled_dot_product_attention(Q, K, V)
print("rows of A add up to 1:", np.allclose(A.sum(axis=1), 1))

# %% [markdown]
# In PyTorch each matrix is an `nn.Linear` without bias. It stores $\mathbf{W}^\top$, so we copy the transposes.

# %%
to_q, to_k, to_v = (nn.Linear(D_MODEL, D_K, bias=False) for _ in range(3))
with torch.no_grad():
    for layer, W in [(to_q, W_Q), (to_k, W_K), (to_v, W_V)]:
        layer.weight.copy_(torch.tensor(W.T))
Q_t = to_q(X_t)
K_t = to_k(X_t)
V_t = to_v(X_t)
output_torch = F.scaled_dot_product_attention(Q_t, K_t, V_t)
print("same result:", np.allclose(output, output_torch.detach().numpy()))

plt.imshow(A, cmap="Blues")
plt.xticks(range(T), sentence, rotation=90)
plt.yticks(range(T), sentence)
plt.xlabel("token being looked at (j)")
plt.ylabel("token that looks (t)")
plt.colorbar(label="weight $a_{tj}$")
plt.show()

# %% [markdown]
# ## 3. Multi-head attention
#
# Each head has its own matrices: $\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_Q^{(i)}, \mathbf{X}\mathbf{W}_K^{(i)}, \mathbf{X}\mathbf{W}_V^{(i)})$.
# The heads go side by side and are mixed: $\text{MultiHead}(\mathbf{X}) = [\text{head}_1, \dots, \text{head}_h]\,\mathbf{W}_O$.
# One big matrix is computed and split into $h$ slices of size $d_k = d_{\text{model}} / h$. The NumPy class mirrors
# `nn.MultiheadAttention(embed_dim, num_heads)`: it returns the outputs and the weights averaged over the heads.

# %%
class MultiheadAttention:
    """NumPy twin of nn.MultiheadAttention without biases, for one sequence."""
    def __init__(self, embed_dim: int, num_heads: int):
        self.num_heads = num_heads
        scale = np.sqrt(embed_dim)
        self.W_Q = rng.normal(size=(embed_dim, embed_dim)) / scale
        self.W_K = rng.normal(size=(embed_dim, embed_dim)) / scale
        self.W_V = rng.normal(size=(embed_dim, embed_dim)) / scale
        self.W_O = rng.normal(size=(embed_dim, embed_dim)) / scale

    def split_heads(self, M: np.ndarray) -> np.ndarray:
        """(T, d_model) -> (h, T, d_k): one slice of columns per head."""
        return M.reshape(len(M), self.num_heads, -1).transpose(1, 0, 2)

    def __call__(self, query, key, value, attn_mask=None):
        Q = self.split_heads(query @ self.W_Q)
        K = self.split_heads(key @ self.W_K)
        V = self.split_heads(value @ self.W_V)
        heads, weights = scaled_dot_product_attention(Q, K, V, attn_mask)  # head_1, ..., head_h
        concatenated = heads.transpose(1, 0, 2).reshape(len(query), -1)  # [head_1, ..., head_h]
        output = concatenated @ self.W_O
        return output, weights.mean(axis=0)


multi_head = MultiheadAttention(D_MODEL, HEADS)
mh_output, mh_weights = multi_head(X, X, X)

# %% [markdown]
# `nn.MultiheadAttention` keeps $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ stacked (and transposed) in `in_proj_weight`
# and $\mathbf{W}_O^\top$ in `out_proj.weight`. With `batch_first=True` the input has shape (batch, $T$, $d_{\text{model}}$).

# %%
torch_mha = nn.MultiheadAttention(D_MODEL, HEADS, bias=False, batch_first=True)
with torch.no_grad():
    stacked = np.vstack([multi_head.W_Q.T, multi_head.W_K.T, multi_head.W_V.T])
    torch_mha.in_proj_weight.copy_(torch.tensor(stacked))
    torch_mha.out_proj.weight.copy_(torch.tensor(multi_head.W_O.T))
batch = X_t.unsqueeze(0)
mh_output_torch, mh_weights_torch = torch_mha(batch, batch, batch)
print("same result:", np.allclose(mh_output, mh_output_torch[0].detach().numpy()))
print("same weights:", np.allclose(mh_weights, mh_weights_torch[0].detach().numpy()))

# %% [markdown]
# ## 4. Adding word order
#
# Attention alone ignores order: shuffling the tokens only shuffles the outputs. So a position vector is added, $\mathbf{x}_t \leftarrow \mathbf{x}_t + \mathbf{p}_t$, with
# $p_{t,2i} = \sin\!\left(t / 10000^{2i/d_{\text{model}}}\right)$ and $p_{t,2i+1} = \cos\!\left(t / 10000^{2i/d_{\text{model}}}\right)$.

# %%
def positional_encoding(length: int, d_model: int) -> np.ndarray:
    """Sinusoidal position vectors, one row per position."""
    t = np.arange(length)[:, None]  # positions as a column
    two_i = np.arange(0, d_model, 2)  # even indices 2i
    angles = t / 10000 ** (two_i / d_model)
    P = np.zeros((length, d_model))
    P[:, 0::2] = np.sin(angles)  # p_{t,2i}
    P[:, 1::2] = np.cos(angles)  # p_{t,2i+1}
    return P


print(positional_encoding(2, 4)[1].round(3))

# %% [markdown]
# PyTorch has no built-in for it; transformer code builds it once with tensor operations.

# %%
t = torch.arange(T).unsqueeze(1)
two_i = torch.arange(0, D_MODEL, 2)
angles = t / 10000 ** (two_i / D_MODEL)
P_t = torch.zeros(T, D_MODEL)
P_t[:, 0::2] = torch.sin(angles)
P_t[:, 1::2] = torch.cos(angles)
print("same result:", np.allclose(positional_encoding(T, D_MODEL), P_t.numpy()))

# %%
order = torch.randperm(T)


def self_attention(inputs: torch.Tensor) -> torch.Tensor:
    """Outputs of the multi-head layer of section 3 for one sequence."""
    batch = inputs.unsqueeze(0)
    outputs, _ = torch_mha(batch, batch, batch)
    return outputs[0]


shuffled_then_attended = self_attention(X_t[order])
attended_then_shuffled = self_attention(X_t)[order]
print("without positions, shuffling only shuffles the outputs:",
      torch.allclose(shuffled_then_attended, attended_then_shuffled))
shuffled_then_attended = self_attention(X_t[order] + P_t)
attended_then_shuffled = self_attention(X_t + P_t)[order]
print("with positions, the outputs change:",
      not torch.allclose(shuffled_then_attended, attended_then_shuffled))

# %% [markdown]
# ## 5. Causal mask and the next token
#
# A decoder must not look ahead: $\mathbf{A} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)$ with
# $M_{tj} = -\infty$ for $j > t$. PyTorch builds the same $\mathbf{M}$ with `nn.Transformer.generate_square_subsequent_mask`,
# and `F.scaled_dot_product_attention(..., is_causal=True)` applies it for us.

# %%
mask = np.triu(np.full((T, T), -np.inf), k=1)
causal_output, causal_A = scaled_dot_product_attention(Q, K, V, mask)
print("no weight on future tokens:", np.all(causal_A[mask == -np.inf] == 0))

torch_mask = nn.Transformer.generate_square_subsequent_mask(T)
print("same mask:", np.array_equal(mask, torch_mask.numpy()))
causal_torch = F.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=True)
print("same result:", np.allclose(causal_output, causal_torch.detach().numpy()))

# %% [markdown]
# A tiny decoder: embedding plus positions, one masked multi-head layer with a residual connection,
# $\mathbf{h}_t = \mathbf{x}_t + \text{MultiHead}(\mathbf{X})_t$, and a linear layer that gives the scores,
# $P(x_{t+1} = w \mid x_1, \dots, x_t) = \text{softmax}(\mathbf{h}_t \mathbf{W} + \mathbf{b})_w$.
# It is trained on one short text with the cross-entropy $J = \frac{1}{T-1}\sum_t -\log P(x_{t+1} = \text{real token})$.

# %%
class TinyDecoder(nn.Module):
    """Embedding + positions, one causal attention layer with a residual, and a linear output."""
    def __init__(self, vocab_size: int, d_model: int, num_heads: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        P = torch.tensor(positional_encoding(length, self.embedding.embedding_dim))
        x = self.embedding(ids) + P  # x_t + p_t
        mask = nn.Transformer.generate_square_subsequent_mask(length)
        attended, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        h = x + attended  # residual: h_t = x_t + MultiHead(X)_t
        scores = self.output(h)  # h_t W + b
        return scores


text = "the cat sat on the mat . the dog sat on the rug .".split()
words = sorted(set(text))
ids = torch.tensor([[words.index(word) for word in text]])
inputs = ids[:, :-1]
next_ids = ids[0, 1:]

model = TinyDecoder(len(words), D_MODEL, HEADS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()
for _ in range(EPOCHS):
    optimizer.zero_grad()
    scores = model(inputs)[0]
    loss = loss_fn(scores, next_ids)
    loss.backward()
    optimizer.step()

# %%
scores_t = model(inputs)[0]
scores = scores_t.detach().numpy()
exp_scores = np.exp(scores)
P_by_hand = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # softmax by rows
P_real = P_by_hand[np.arange(len(next_ids)), next_ids.numpy()]  # P(x_{t+1} = real token)
J_by_hand = -np.mean(np.log(P_real))
J_torch = loss_fn(scores_t, next_ids).item()
print(f"cross-entropy {J_by_hand:.3f} | same result:", np.isclose(J_by_hand, J_torch))
for t in [1, 4]:
    context = " ".join(text[:t + 1])
    predicted = words[P_by_hand[t].argmax()]
    print(f"after '{context}' the model predicts '{predicted}'")

# %% [markdown]
# ## Exercises
#
# 1. Give section 2 key size 64 (`D_K = 64`) and compare the largest weight in each row of $\mathbf{A}$ with
#    `F.scaled_dot_product_attention(..., scale=1.0)`, which skips the division by $\sqrt{d_k}$, and with the default scale.
# 2. Divide the scores of the tiny decoder by a temperature $\tau$ (0.5, 1, 2) before `torch.softmax` and watch how sharp the probabilities get.
# 3. Plot `positional_encoding(50, 16)` with `plt.imshow` and find which entries change fastest with the position.
