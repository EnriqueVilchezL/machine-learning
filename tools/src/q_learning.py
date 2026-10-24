# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Q-learning
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/EnriqueVilchezL/machine-learning/blob/main/book/05-reinforcement/q-learning/q-learning-notebook.ipynb)
#
# Companion notebook for the *Q-learning* slides. Each idea is written first by hand with NumPy
# and then checked with an independent computation that should give the same answer.
#
# **Data:** no data set; the agent moves in the 4 × 3 grid world (wall at (2, 2), goal +1, pit −1, each move −0.04), here without slipping.

# %%
import matplotlib.pyplot as plt
import numpy as np

RANDOM_STATE = 42
GAMMA = 0.9
ALPHA = 0.5
EPSILON = 0.2
EPISODES = 3000
rng = np.random.default_rng(RANDOM_STATE)


class GridWorld:
    """The 4 × 3 grid world: wall at (2, 2), goal +1 at (4, 3), pit −1 at (4, 2), start at (1, 1)."""

    cells = [(x, y) for y in range(3) for x in range(4) if (x, y) != (1, 1)]  # (column, row), 0-based
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left

    def __init__(self, slip=0.2, step_reward=-0.04, random_state=None):
        self.slip, self.step_reward = slip, step_reward
        self.rng = np.random.default_rng(random_state)
        self.n_states, self.n_actions = len(self.cells), len(self.moves)
        self.start, self.goal, self.pit = (self.cells.index(c) for c in [(0, 0), (3, 2), (3, 1)])
        self.terminal = [self.goal, self.pit]
        self.P, self.R = self._model()

    def _move(self, s, direction):
        """Next cell when moving in a direction; hitting a wall means staying."""
        cell = (self.cells[s][0] + self.moves[direction][0], self.cells[s][1] + self.moves[direction][1])
        return self.cells.index(cell) if cell in self.cells else s

    def _model(self):
        """Chances P[s, a, s'] and rewards R[s, a, s']; a terminal state stays put with reward 0."""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s in self.terminal:
                    P[s, a, s] = 1
                    continue
                for direction, chance in [(a, 1 - self.slip), ((a + 1) % 4, self.slip / 2), ((a + 3) % 4, self.slip / 2)]:
                    P[s, a, self._move(s, direction)] += chance
        R = np.full(P.shape, self.step_reward)
        R[:, :, self.goal], R[:, :, self.pit] = 1, -1
        R[self.terminal] = 0
        return P, R

    def reset(self, state=None):
        """Start an episode in `state` (the start cell by default) and return it."""
        self.state = self.start if state is None else state
        return self.state

    def step(self, action):
        """Sample the next state from P and return (next_state, reward, done)."""
        next_state = self.rng.choice(self.n_states, p=self.P[self.state, action])
        reward = self.R[self.state, action, next_state]
        self.state = next_state
        return next_state, reward, next_state in self.terminal


env = GridWorld(slip=0.0, random_state=RANDOM_STATE)  # the agent only calls reset and step, never reads P or R

# %% [markdown]
# ## 1. Learning in small steps
#
# A running mean $\bar{x}_n = \bar{x}_{n-1} + \frac{1}{n}(x_n - \bar{x}_{n-1})$ never stores the old numbers,
# yet ends at the ordinary mean.

# %%
numbers = rng.normal(5, 2, size=1000)
mean = 0.0
for n, x in enumerate(numbers, start=1):
    mean += (x - mean) / n

print(f"running mean {mean:.4f} | same result:", np.isclose(mean, np.mean(numbers)))

# %% [markdown]
# ## 2. The Q-learning update
#
# Target $y = r + \gamma \max_{a'} Q(s', a')$, TD error $\delta = y - Q(s, a)$, update $Q(s, a) \leftarrow Q(s, a) + \alpha\,\delta$.
# The agent below keeps the table in `q_values_`; `fit` plays episodes, choosing actions ε-greedily with `act`
# and applying `update` after every step, and `predict` returns the greedy action.

# %%
class QLearning:
    """Tabular Q-learning agent with ε-greedy exploration."""

    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.2, n_episodes=3000, random_state=None):
        self.alpha, self.gamma, self.epsilon, self.n_episodes = alpha, gamma, epsilon, n_episodes
        self.rng = np.random.default_rng(random_state)

    def predict(self, state):
        return np.argmax(self.q_values_[state], axis=-1)

    def act(self, state):
        """ε-greedy: a random action with chance ε, otherwise the greedy one."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.q_values_.shape[1])
        return self.predict(state)

    def update(self, state, action, reward, next_state, done):
        """Move Q(s, a) a fraction α toward the target; a terminal next state is worth 0."""
        target = reward if done else reward + self.gamma * self.q_values_[next_state].max()
        self.q_values_[state, action] += self.alpha * (target - self.q_values_[state, action])

    def fit(self, env):
        self.q_values_ = np.zeros((env.n_states, env.n_actions))
        starts = [s for s in range(env.n_states) if s not in env.terminal]
        self.history_ = []
        for _ in range(self.n_episodes):
            state, done = env.reset(self.rng.choice(starts)), False
            while not done:
                action = self.act(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
            self.history_.append(self.q_values_.copy())
        return self


# %% [markdown]
# The deck's example: $Q(s, a) = 0.5$, $r = 0$, $\max_{a'} Q(s', a') = 0.8$, $\alpha = 0.1$. Moving a fraction
# $\alpha$ toward the target is the same as the blend $(1 - \alpha)\,Q + \alpha\, y$.

# %%
q_old, r, max_next, alpha = 0.5, 0.0, 0.8, 0.1
target = r + GAMMA * max_next
agent = QLearning(alpha=alpha, gamma=GAMMA)
agent.q_values_ = np.array([[q_old, 0, 0, 0],      # state s
                            [max_next, 0, 0, 0]])  # state s'
agent.update(state=0, action=0, reward=r, next_state=1, done=False)

print(f"y = {target:.2f}, delta = {target - q_old:.2f}, new Q = {agent.q_values_[0, 0]:.3f}")
print("same result:", np.isclose(agent.q_values_[0, 0], (1 - alpha) * q_old + alpha * target))

# %% [markdown]
# ## 3. ε-greedy exploration
#
# With chance $\varepsilon$ take a random action, otherwise the greedy one, so the greedy action has chance
# $1 - \varepsilon + \varepsilon/|\mathcal{A}|$ and each other action $\varepsilon/|\mathcal{A}|$.
# Many calls to `act` should match these chances (up to sampling noise).

# %%
agent = QLearning(epsilon=EPSILON, random_state=RANDOM_STATE)
agent.q_values_ = np.array([[0.66, 0.71, 0.55, 0.60]])  # one state, actions ↑ → ↓ ←
chances = np.full(env.n_actions, EPSILON / env.n_actions)
chances[agent.predict(0)] += 1 - EPSILON

choices = [agent.act(0) for _ in range(20_000)]
print("formula:", chances, "| same result:", np.allclose(np.bincount(choices) / len(choices), chances, atol=0.01))

# %% [markdown]
# ## 4. Q-learning on the grid
#
# For reference, value iteration with the model (Topic 20) gives the exact
# $Q^*(s, a) = \sum_{s'} P(s' \mid s, a)\big[R(s, a, s') + \gamma \max_{a'} Q^*(s', a')\big]$.
# The learner never uses it: it only sees tuples $(s, a, r, s')$ from `step`.
# Each episode starts in a random cell so that every pair keeps being tried.

# %%
Q_star = np.zeros((env.n_states, env.n_actions))
for _ in range(200):
    Q_star = (env.P * (env.R + GAMMA * Q_star.max(axis=1))).sum(axis=2)

# %%
agent = QLearning(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, n_episodes=EPISODES, random_state=RANDOM_STATE).fit(env)

errors = np.abs(np.array(agent.history_) - Q_star).max(axis=(1, 2))
plt.semilogy(errors)
plt.xlabel("episode")
plt.ylabel("largest error  $|Q - Q^*|$")
plt.show()

# %% [markdown]
# Learned values are approximate, so we compare them with a small tolerance, and we check that the greedy
# action is optimal in every cell (when two actions tie, either one counts).

# %%
non_terminal = [s for s in range(env.n_states) if s not in env.terminal]
greedy = agent.predict(non_terminal)
print("same values:", np.allclose(agent.q_values_[non_terminal], Q_star[non_terminal], atol=0.01))
print("greedy policy is optimal:", np.allclose(Q_star[non_terminal, greedy], Q_star[non_terminal].max(axis=1)))

# %% [markdown]
# ## Exercises
#
# 1. Set `EPSILON = 0` and start every episode in cell (1, 1). Which entries of `q_values_` stay at 0?
# 2. Write SARSA by changing `update` to use the target $r + \gamma\, Q(s', a')$, with $a'$ the action really taken next, and compare the greedy policies.
# 3. Try `ALPHA = 0.05` and `ALPHA = 1`. How does the error curve of section 4 change?
