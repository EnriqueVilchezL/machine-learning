---
title: Machine Learning
short_title: Home
description: Lecture slides for an introductory machine learning course, one page per topic.
---

```{iframe} assets/hero.html
:width: 100%
```

## Why machine learning

Your phone unlocks when it sees your face. Your bank flags a card payment made in another country a few minutes after you bought lunch at home. Nobody wrote an `if` statement for your face or for your spending habits. Both systems learned those rules from examples.

That's the idea this course is built on. You collect data, pick a model, and let an algorithm adjust the model until its predictions are good. The animation above does exactly that with a small neural network, live in your browser.

```{epigraph}
A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

-- Tom M. Mitchell, *Machine Learning* (1997)
```

In the animation, the task T is labeling points in the plane, the experience E is the 240 labeled examples on the left, and the measure P is the accuracy you see climb at the bottom right.

## What's in the course

::::{grid} 1 1 2 2

:::{card} 1. Introduction
What it means for a program to learn from data, and how to check that it did.

- [Introduction to machine learning](01-intro/intro-ml/intro-ml.md)
+++
1 topic
:::

:::{card} 2. Data preparation
From raw data to data a model can use, and how to measure results honestly.

- [Data cleaning](02-data-preparation/data-cleaning/data-cleaning.md)
- [Feature engineering](02-data-preparation/feature-engineering/feature-engineering.md)
- [Normalization](02-data-preparation/normalization/normalization.md)
- [Train-test split](02-data-preparation/train-test-split/train-test-split.md)
- [Cross-validation](02-data-preparation/cross-validation/cross-validation.md)
+++
5 topics
:::

:::{card} 3. Supervised learning
Models that learn from labeled examples.

- [Supervised learning](03-supervised/supervised-learning/supervised-learning.md)
- [k-nearest neighbors](03-supervised/knn/knn.md)
- [Regression](03-supervised/regression/regression.md)
- [Decision trees](03-supervised/decision-trees/decision-trees.md)
- [Logistic regression](03-supervised/logistic-regression/logistic-regression.md)
- [Support vector machines](03-supervised/svm/svm.md)
- [Naive Bayes](03-supervised/naive-bayes/naive-bayes.md)
- [Ensemble methods](03-supervised/ensembles/ensembles.md)
+++
8 topics
:::

:::{card} 4. Unsupervised learning
Structure in data without labels: groups of similar examples and fewer, more informative features.

- [Unsupervised learning](04-unsupervised/unsupervised-learning/unsupervised-learning.md)
- [k-means clustering](04-unsupervised/kmeans/kmeans.md)
- [Hierarchical clustering](04-unsupervised/hierarchical/hierarchical.md)
- [Principal component analysis](04-unsupervised/pca/pca.md)
+++
4 topics
:::

:::{card} 5. Reinforcement learning
Agents that learn by trial, error and reward.

- [Reinforcement learning](05-reinforcement/reinforcement-learning/reinforcement-learning.md)
- [Markov decision processes](05-reinforcement/mdp/mdp.md)
- [Q-learning](05-reinforcement/q-learning/q-learning.md)
+++
3 topics
:::

:::{card} 6. Deep learning
Neural networks with many layers, and the architectures behind images, text and games.

- [Deep learning](06-deep-learning/deep-learning/deep-learning.md)
- [Neural networks](06-deep-learning/neural-networks/neural-networks.md)
- [Convolutional neural networks](06-deep-learning/cnn/cnn.md)
- [Recurrent neural networks](06-deep-learning/rnn/rnn.md)
- [Autoencoders](06-deep-learning/autoencoders/autoencoders.md)
- [Generative adversarial networks](06-deep-learning/gan/gan.md)
- [Transfer learning](06-deep-learning/transfer-learning/transfer-learning.md)
- [Deep reinforcement learning](06-deep-learning/deep-rl/deep-rl.md)
- [Transformers](06-deep-learning/transformers/transformers.md)
+++
9 topics
:::

::::

## Course roadmap

```{mermaid}
flowchart TB
  subgraph base [Foundations]
    direction LR
    A[1. Introduction] --> B[2. Data preparation]
  end
  subgraph kinds [Three ways to learn]
    direction LR
    C[3. Supervised learning] --> D[4. Unsupervised learning] --> E[5. Reinforcement learning]
  end
  subgraph deep [Going deeper]
    direction LR
    F[6. Deep learning]
  end
  base --> kinds --> deep
  classDef ready fill:#ccfbf1,stroke:#0d9488,color:#0f172a
  class A,B,C,D,E,F ready
  style base fill:#ffffff,stroke:#e2e8f0
  style kinds fill:#ffffff,stroke:#e2e8f0
  style deep fill:#ffffff,stroke:#e2e8f0
```

## Using the slides

Every topic page holds one slide deck. The slides show the theory with figures and step-by-step animations, and some slides have buttons or sliders you can play with.

```{tip} Keyboard shortcuts
Click inside a deck first. Then {kbd}`→` or {kbd}`Space` moves to the next slide, {kbd}`←` goes back, and {kbd}`Home` or {kbd}`End` jumps to the first or last slide.
```

:::{dropdown} Background that helps
You'll follow the slides more easily if you're comfortable with vectors and matrices, derivatives, and basic probability (mean, variance, conditional probability). Each deck introduces the notation it uses, so you can look things up as they come.
:::
