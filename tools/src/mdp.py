# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Markov decision processes
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/05-reinforcement/mdp/mdp-notebook.ipynb)
#
# Companion notebook for the *Markov decision processes* slides. Each idea is written first by hand
# with NumPy and then checked with an independent computation that should give the same answer.
#
# **Data:** no data set; the agent moves in the slippery 4 × 3 grid world (wall at (2, 2), goal +1, pit −1, each step −0.04).

# %%
import matplotlib.pyplot as plt
import numpy as np

RANDOM_STATE = 42
GAMMA = 0.9
STEP_REWARD = -0.04
rng = np.random.default_rng(RANDOM_STATE)

CELLS = [(x, y) for y in range(3) for x in range(4) if (x, y) != (1, 1)]  # (column, row), 0-based
MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left
ARROWS = "↑→↓←"
GOAL, PIT, START = CELLS.index((3, 2)), CELLS.index((3, 1)), CELLS.index((0, 0))
N_STATES, N_ACTIONS = len(CELLS), len(MOVES)


def move(s, direction):
    """Next cell when moving in a direction; hitting a wall means staying."""
    cell = (CELLS[s][0] + MOVES[direction][0], CELLS[s][1] + MOVES[direction][1])
    return CELLS.index(cell) if cell in CELLS else s


P = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # P[s, a, s']
for s in range(N_STATES):
    for a in range(N_ACTIONS):
        if s in (GOAL, PIT):
            P[s, a, s] = 1  # terminal: stay forever with reward 0
            continue
        for direction, chance in [(a, 0.8), ((a + 1) % 4, 0.1), ((a + 3) % 4, 0.1)]:
            P[s, a, move(s, direction)] += chance

R = np.full((N_STATES, N_ACTIONS, N_STATES), STEP_REWARD)  # R[s, a, s']
R[:, :, GOAL], R[:, :, PIT] = 1, -1
R[[GOAL, PIT]] = 0

# %% [markdown]
# ## 1. The transition chances
#
# One table $P(s' \mid s, a)$ describes all the dynamics, and each row is a distribution:
# $\sum_{s'} P(s' \mid s, a) = 1$. Sampling many moves should give the same chances (up to sampling noise).

# %%
print("every row sums to 1:", np.allclose(P.sum(axis=2), 1))

UP = 0
samples = rng.choice(N_STATES, size=10_000, p=P[START, UP])
frequencies = np.bincount(samples, minlength=N_STATES) / len(samples)
for s in np.flatnonzero(P[START, UP]):
    print(f"start, up -> cell {CELLS[s][0] + 1, CELLS[s][1] + 1}: chance {P[START, UP, s]:.1f}")
print("same result:", np.allclose(frequencies, P[START, UP], atol=0.02))

# %% [markdown]
# ## 2. The return
#
# $G_t = \sum_k \gamma^k R_{t+k+1} = R_{t+1} + \gamma\, G_{t+1}$. The deck's robot episode with rewards +2, +2, −3:
# the direct sum and the backward recursion agree.

# %%
rewards = np.array([2, 2, -3])
return_by_sum = np.sum(GAMMA ** np.arange(len(rewards)) * rewards)

return_by_recursion = 0
for r in rewards[::-1]:
    return_by_recursion = r + GAMMA * return_by_recursion

print(f"G_0 = {return_by_sum:.2f} | same result:", np.isclose(return_by_sum, return_by_recursion))

# %% [markdown]
# ## 3. Policy evaluation
#
# For a fixed policy $\pi(s)$ the Bellman equation is
# $V^\pi(s) = \sum_{s'} P(s' \mid s, \pi(s)) \big[R(s, \pi(s), s') + \gamma V^\pi(s')\big]$.
# Sweeping it from $V = 0$ converges; since it is linear in $V^\pi$, `np.linalg.solve` also gives it exactly.

# %%
policy = np.array([0, 3, 0, 3,  # bottom row: up, left, up, left
                   0, 0, 0,     # middle row: up, up, (pit)
                   1, 1, 1, 0])  # top row: right, right, right, (goal)
rows = np.arange(N_STATES)
P_pi, R_pi = P[rows, policy], R[rows, policy]

V = np.zeros(N_STATES)
while True:
    V_new = (P_pi * (R_pi + GAMMA * V)).sum(axis=1)
    if np.max(np.abs(V_new - V)) < 1e-10:
        break
    V = V_new

V_solve = np.linalg.solve(np.eye(N_STATES) - GAMMA * P_pi, (P_pi * R_pi).sum(axis=1))
print(f"V(start) = {V[START]:.3f} | same result:", np.allclose(V, V_solve))

# %% [markdown]
# ## 4. Value iteration and policy iteration
#
# Value iteration repeats $V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a)\big[R(s, a, s') + \gamma V_k(s')\big]$
# and reads the greedy policy $\pi^*(s) = \arg\max_a Q^*(s, a)$.
# Policy iteration (evaluate with `solve`, then act greedily, repeat) should reach the same $V^*$ and $\pi^*$.

# %%
V_star = np.zeros(N_STATES)
while True:
    Q = (P * (R + GAMMA * V_star)).sum(axis=2)
    if np.max(np.abs(Q.max(axis=1) - V_star)) < 1e-10:
        break
    V_star = Q.max(axis=1)
policy_vi = Q.argmax(axis=1)

policy_pi = np.full(N_STATES, 3)  # start from "always go left"
while True:
    P_pi, R_pi = P[rows, policy_pi], R[rows, policy_pi]
    V_pi = np.linalg.solve(np.eye(N_STATES) - GAMMA * P_pi, (P_pi * R_pi).sum(axis=1))
    improved = (P * (R + GAMMA * V_pi)).sum(axis=2).argmax(axis=1)
    if np.array_equal(improved, policy_pi):
        break
    policy_pi = improved

non_terminal = [s for s in range(N_STATES) if s not in (GOAL, PIT)]
print("same values:", np.allclose(V_star, V_pi))
print("same policy:", np.array_equal(policy_vi[non_terminal], policy_pi[non_terminal]))

# %%
grid = np.full((3, 4), np.nan)
for s, (x, y) in enumerate(CELLS):
    grid[y, x] = V_star[s]
    label = {GOAL: "goal +1", PIT: "pit −1"}.get(s, f"{ARROWS[policy_vi[s]]}\n{V_star[s]:.2f}")
    plt.text(x, y, label, ha="center", va="center")
plt.imshow(grid, origin="lower", cmap="coolwarm_r", vmin=-1, vmax=1)
plt.xticks(range(4), range(1, 5))
plt.yticks(range(3), range(1, 4))
plt.title("optimal values $V^*$ and greedy policy (wall in white)")
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. Set `STEP_REWARD = -2` and rerun section 4. Why does the agent now head for the pit from some cells?
# 2. Evaluate the policy "always go right" with `np.linalg.solve` and compare its $V(\text{start})$ with $V^*(\text{start})$.
# 3. Count how many sweeps value iteration needs for $\gamma = 0.5$ and for $\gamma = 0.99$.
