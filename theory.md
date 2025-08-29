# Tennis as a Hierarchical Markov Chain: Mathematical Theory and Results

## Table of Contents

1. [Introduction](#introduction)
2. [Markov Chain Fundamentals](#markov-chain-fundamentals)
3. [Absorbing Markov Chains](#absorbing-markov-chains)
4. [The Fundamental Matrix](#the-fundamental-matrix)
5. [Tennis Game Structure](#tennis-game-structure)
6. [Game Level Analysis](#game-level-analysis)
7. [Set Level Analysis](#set-level-analysis)
8. [Match Level Analysis](#match-level-analysis)
9. [Results and Interpretations](#results-and-interpretations)
10. [Extensions and Future Work](#extensions-and-future-work)
11. [Conclusion](#conclusion)
12. [References](#references)

---

## Introduction

I have played tennis for a long time and also been a fan. Tennis like many other sports is heavy in statistics, and recently I watched Djokovic and Alcaraz play a Grand Slam final, where towards the end of a very tight match, they had both won as many points. This got me thinking how can we have such tight matches where the total point difference is so marginal? And more importantly, how does the difference in total points reflect on the result of the match? This is equivalent to saying how can we map the probability of a player winning a point to the probability of a player winning the match?

In order to answer these questions, I used a mathematical model called Markov Chains which models systems evolving over time in this case in discrete space where the next state only depends on the current state. In tennis, this means that the possible scores of the next point depend on the current score alone. Furthermore, we will simply use a fixed probability $p$ to denote the likelihood that player 1 wins the point and $1-p$ denotes the probability of player 2 winning the point. This is obviously a strong assumption and reality is not exactly like this ($p$ in reality can depend on the fitness of the player, weather, court surface, mental state, strategy, etc.), but this model can still give us solid insights that will answer our questions.

Through this analysis, we will see that tennis exhibits a remarkable mathematical property: small advantages in winning individual points compound dramatically through the game's hierarchical structure. This is an amplification effect, where slight advantages in winning individual points give larger probabilities of winning the match.

### The Tennis Point System

Tennis has four distinct levels, each building upon the previous:

```
Points → Games → Sets → Matches
```
In a game, points are counted 0, 15, 30, 40. The player that wins 4 points wins the game. If tied at 40-40, the game is decided by difference of two points.

A set is won when a player reaches 6 games. Unless the players tie at 5-5, where the set is won by the first player to reach 7 games won. If tied at 6-6, a 7-point tiebreak is played to determine the winner of the set.

Through this structure of the game, we will start with a probability of winning a point and derive the probability of winning the game. Then we will use that to derive the probability of winning a set, and then use that to estimate the probability of winning the match.

### Mathematical Framework

We model each level (game, set, match) as an **absorbing Markov chain** which is a Markov chain (MC) that has at least one terminal/absorbing state in which it stays forever after. In tennis, each game, set, and match has two terminal states: either Player 1 won or lost. In our MCs:
- States represent the current score.
- Transition probabilities depend on the point win probability.
- Absorbing states represent completed games/sets/matches.

---

## Markov Chain Fundamentals

### Definition

A **Markov chain** is a stochastic process where the probability of future states depends only on the current state, not on the sequence of events that led to it.

Formally, for a sequence of random variables $X_0, X_1, X_2, ...$:

$$P(X_{n+1} = j | X_n = i, X_{n-1} = i_{n-1}, ..., X_0 = i_0) = P(X_{n+1} = j | X_n = i)$$

### Transition Matrix

For a Markov chain with states $\{1, 2, ..., N\}$, the **transition matrix** $\mathbf{P}$ is defined as:

$$P_{ij} = P(X_{n+1} = j | X_n = i)$$

where $P_{ij}$ is the probability of transitioning from state $i$ to state $j$.
MCs have the following properties:

1. $P_{ij} \geq 0$ for all $i, j$
2. $\sum_{j=1}^N P_{ij} = 1$ for all $i$
3. $\mathbf{P}^{(n)} = \mathbf{P}^n$ where the entries of $\mathbf{P}^{(n)}$ are the probabilities of starting at state $i$ and ending in state $j$ after $n$ steps, $P^{(n)}_{ij} = P(X_{n} = j | X_0 = i)$.

---

## Absorbing Markov Chains

### Definition

An **absorbing state** is a state where $P_{ii} = 1$ and $P_{ij} = 0$ for all $j \neq i$ for some row $i$. Once entered, the process remains there forever.

An **absorbing Markov chain** is one that:
1. Has at least one absorbing state
2. From every non-absorbing state, it's possible to reach an absorbing state

The tennis Markov Chain is an absorbing one, as the games, sets, and matches have a ending states: either player 1 wins or player 2 wins.

### Canonical Form

Given the fact that in any absorbing MC, $P_{ii} = 1$ and $P_{ij} = 0$ for all $j \neq i$ for some row $i$, we can rewrite the transition matrix in its canonical form:

$$\mathbf{P} = \begin{pmatrix} \mathbf{Q} & \mathbf{R} \\ \mathbf{0} & \mathbf{I} \end{pmatrix}$$

where:
- $\mathbf{Q}$ is $t \times t$ (transient to transient transitions)
- $\mathbf{R}$ is $t \times a$ (transient to absorbing transitions)
- $\mathbf{0}$ is $a \times t$ (absorbing to transient)
- $\mathbf{I}$ is the identity matrix $a \times a$ (absorbing to absorbing - stay forever)

Here, $t$ = number of transient states, $a$ = number of absorbing states.

Writing the transition matrix in this form will be useful for us as we determine the probabilities of ending in each of the absorbing states.

---

## The Fundamental Matrix

### Derivation

Let $N_{ij}$ be the expected number of times the process visits transient state $j$, starting from transient state $i$. $N_{ij}$ defines the matrix $N$.

From state $i$, we can visit state $j$ in two ways:
1. Directly (if $i = j$): contributes 1 if $i = j$, 0 otherwise
2. Via intermediate states: $\sum_{k} Q_{ik} \cdot N_{kj}$

This gives us the fundamental equation:

$$N_{ij} = \delta_{ij} + \sum_{k=1}^t Q_{ik} N_{kj}$$

where $\delta_{ij}$ is the Kronecker delta (1 if $i = j$, 0 otherwise).

In matrix form:
$$\mathbf{N} = \mathbf{I} + \mathbf{Q}\mathbf{N}$$

Solving for $\mathbf{N}$:
$$\mathbf{N} - \mathbf{Q}\mathbf{N} = \mathbf{I}$$
$$(\mathbf{I} - \mathbf{Q})\mathbf{N} = \mathbf{I}$$
$$\mathbf{N} = (\mathbf{I} - \mathbf{Q})^{-1}$$

This requires $P^k$ to have all non-zero entries for some natural number $k$.

### Absorption Probabilities

The **absorption probability matrix** $\mathbf{B}$ gives the probability of eventual absorption into each absorbing state, starting from each transient state.

$$B_{ij} = \sum_{k=1}^t N_{ik} R_{kj}$$

In matrix form:
$$\mathbf{B} = \mathbf{N}\mathbf{R} = (\mathbf{I} - \mathbf{Q})^{-1}\mathbf{R}$$

where
- $N_{ij}$: Expected number of visits to transient state $j$ starting from state $i$.
- $B_{ij}$: Probability of eventually being absorbed into absorbing state $j$ starting from transient state $i$.

---

## Tennis Game Structure

### Scoring System

Tennis uses a unique scoring system at each level:

**Points within a Game:**
- 0, 15, 30, 40 (represented as 0, 1, 2, 3)
- Special rules for deuce (40-40 or 3-3) requiring a 2-point margin

**Games within a Set:**
- First to 6 games, but must win by 2
- Tiebreak at 6-6 (simplified in our model)

**Sets within a Match:**
- Best of 3 or best of 5 sets
- First to win majority of sets

### State Space Design

For computational efficiency and clarity, we represent states as strings:

**Game Level:**
- Regular states: "0-0", "1-0", "2-1", etc.
- Special states: "Deuce", "Adv-P1", "Adv-P2"
- Terminal states: "P1-Wins", "P2-Wins"

**Set Level:**
- Regular states: "0-0", "3-2", "5-4", etc.
- Special states: "Tiebreak"
- Terminal states: "P1-Set", "P2-Set"

**Match Level:**
- Regular states: "0-0", "1-0", "1-1", etc.
- Terminal states: "P1-Match", "P2-Match"

---

## Game Level Analysis

### State Space

For a game, we define states $S_{\text{game}} = \{(i,j) : 0 \leq i,j \leq 3\} \cup \{\text{Deuce}, \text{Adv-P1}, \text{Adv-P2}, \text{P1-Wins}, \text{P2-Wins}\}$

### Transition Rules

Given point win probability $p$ for Player 1:

**Regular States $(i,j)$ where $i,j < 3$:**
- With probability $p$: $(i,j) \rightarrow (i+1,j)$
- With probability $1-p$: $(i,j) \rightarrow (i,j+1)$

**Winning States:**
- $(3,j)$ where $j < 3$: $p \rightarrow$ P1-Wins, $1-p \rightarrow (3,j+1)$
- $(i,3)$ where $i < 3$: $p \rightarrow (i+1,3)$, $1-p \rightarrow$ P2-Wins

**Deuce Transitions:**
- $(3,3) \rightarrow$ Deuce
- Deuce: $p \rightarrow$ Adv-P1, $1-p \rightarrow$ Adv-P2
- Adv-P1: $p \rightarrow$ P1-Wins, $1-p \rightarrow$ Deuce
- Adv-P2: $p \rightarrow$ Deuce, $1-p \rightarrow$ P2-Wins

### Mathematical Model

The transition matrix $\mathbf{P}_{\text{game}}$ is constructed by encoding these rules. For example:

$$P_{\text{game}}[\text{"0-0"}, \text{"1-0"}] = p$$
$$P_{\text{game}}[\text{"0-0"}, \text{"0-1"}] = 1-p$$
$$P_{\text{game}}[\text{"Deuce"}, \text{"Adv-P1"}] = p$$

The full transition matrix is then given by:

$$
\mathbf{P} = \begin{pmatrix}
 & \text{0-0} & \text{0-1} & \text{0-2} & \text{0-3} & \text{1-0} & \text{1-1} & \text{1-2} & \text{1-3} & \text{2-0} & \text{2-1} & \text{2-2} & \text{2-3} & \text{3-0} & \text{3-1} & \text{3-2} & \text{3-3} & \text{Deuce} & \text{Adv-P1} & \text{Adv-P2} & \text{P1-W} & \text{P2-W} \\
\text{0-0} & 0 & 1-p & 0 & 0 & p & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\text{0-1} & 0 & 0 & 1-p & 0 & 0 & p & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\text{0-2} & 0 & 0 & 0 & 1-p & 0 & 0 & p & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\text{0-3} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & p & 1-p \\
\text{1-0} & 0 & 0 & 0 & 0 & 0 & 1-p & 0 & 0 & p & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\text{1-1} & 0 & 0 & 0 & 0 & 0 & 0 & 1-p & 0 & 0 & p & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\text{1-2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1-p & 0 & 0 & p & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\text{1-3} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & p & 1-p \\
\text{2-0} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1-p & 0 & 0 & p & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\text{2-1} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1-p & 0 & 0 & p & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\text{2-2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1-p & 0 & 0 & p & 0 & 0 & 0 & 0 & 0 & 0 \\
\text{2-3} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & p & 0 & 0 & 0 & 1-p \\
\text{3-0} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & p & 1-p \\
\text{3-1} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & p & 1-p \\
\text{3-2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1-p & 0 & 0 & p & 0 \\
\text{3-3} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
\text{Deuce} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & p & 1-p & 0 & 0 \\
\text{Adv-P1} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1-p & 0 & 0 & p & 0 \\
\text{Adv-P2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1-p & 0 & 0 & 0 & p \\
\text{P1-W} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
\text{P2-W} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

### Game Win Probability

Applying the fundamental matrix method:

$$\mathbf{N}_{\text{game}} = (\mathbf{I} - \mathbf{Q}_{\text{game}})^{-1}$$

$$\mathbf{B}_{\text{game}} = \mathbf{N}_{\text{game}}\mathbf{R}_{\text{game}}$$

The game win probability for Player 1 starting from "0-0" is:

$$P(\text{Player 1 wins game}) = B_{\text{game}}[\text{"0-0"}, \text{"P1-Wins"}]$$


$$
p_g:= P(\text{Player 1 wins game}) = p^{4} \left(- 20 p^{3} + 70 p^{2} - 84 p + 35\right)
$$

---

## Set Level Analysis

### State Space

$$S_{\text{set}} = \{(i,j) : 0 \leq i,j \leq 6, \text{valid tennis score}\} \cup \{\text{Tiebreak}, \text{P1-Set}, \text{P2-Set}\}$$

where valid tennis scores exclude impossible combinations like $(7,3)$.

### Transition Rules

Given game win probability $p_g$ for Player 1:

**Regular States $(i,j)$:**
- If $i < 6$ and $j < 6$: Normal progression
- If $i = 5, j = 5$: Can go to $(6,5)$, $(5,6)$, then to set win or $(6,6)$
- If $i = 6, j = 6$: Must go to tiebreak

**Set Win Conditions:**
- Win by reaching 6 with margin ≥ 2
- Win tiebreak when at 6-6

### Set Win Probability

$$P_{\text{set}}(p_g) = B_{\text{set}}[\text{"0-0"}, \text{"P1-Set"}]$$

is given by $$p_s:= P(\text{Player 1 wins set})$$


$$
\begin{align}
p_s &= p_{g}^{6} \bigg(504 p_{g}^{7} - 3276 p_{g}^{6} + 8820 p_{g}^{5} \\
&\qquad - 12474 p_{g}^{4} + 9520 p_{g}^{3} - 3339 p_{g}^{2} \\
&\qquad + 36 p_{g} + 210\bigg) \\

&= p^{24} \left(- 20 p^{3} + 70 p^{2} - 84 p + 35\right)^{6} \\
&\quad \times \bigg[ 504 p^{28} \left(- 20 p^{3} + 70 p^{2} - 84 p + 35\right)^{7} \\
&\qquad\qquad - 3276 p^{24} \left(- 20 p^{3} + 70 p^{2} - 84 p + 35\right)^{6} \\
&\qquad\qquad + 8820 p^{20} \left(- 20 p^{3} + 70 p^{2} - 84 p + 35\right)^{5} \\
&\qquad\qquad - 12474 p^{16} \left(- 20 p^{3} + 70 p^{2} - 84 p + 35\right)^{4} \\
&\qquad\qquad + 9520 p^{12} \left(- 20 p^{3} + 70 p^{2} - 84 p + 35\right)^{3} \\
&\qquad\qquad - 3339 p^{8} \left(- 20 p^{3} + 70 p^{2} - 84 p + 35\right)^{2} \\
&\qquad\qquad + 36 p^{4} \left(- 20 p^{3} + 70 p^{2} - 84 p + 35\right) \\
&\qquad\qquad + 210 \bigg]
\end{align}
$$


---

## Match Level Analysis

### State Space

For best-of-$n$ match (where $n \in \{3,5\}$):
$$S_{\text{match}} = \{(i,j) : 0 \leq i,j \leq \lceil n/2 \rceil, i < \lceil n/2 \rceil \text{ or } j < \lceil n/2 \rceil\} \cup \{\text{P1-Match}, \text{P2-Match}\}$$

### Transition Rules

Given set win probability $p_s$ for Player 1:
- From $(i,j)$ where $i,j < \lceil n/2 \rceil$:
  - With probability $p_s$: $(i,j) \rightarrow (i+1,j)$
  - With probability $1-p_s$: $(i,j) \rightarrow (i,j+1)$

### Match Win Probability

$$P_{\text{match}}(p_s) = B_{\text{match}}[\text{"0-0"}, \text{"P1-Match"}]$$

If the match is best of 3, the probability $$p_m^{(3)}:= P(\text{Player 1 wins Bo3})$$ is given by:

$$
p_m^{(3)} =
p_{s}^{2} \left(3 - 2 p_{s}\right)
$$

In terms of $p$, we get:


$$
\begin{align}
p_m^{(3)} &= p^{48} \bigg(- 2 p^{24} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{6} \big(504 p^{28} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{7} \\
&\quad - 3276 p^{24} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{6}
+ 8820 p^{20} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{5} \\
&\quad - 12474 p^{16} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{4}
 + 9520 p^{12} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{3} \\
&\quad - 3339 p^{8} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{2} + 36 p^{4} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big) + 210\big) \\
&\quad + 3\big) \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{12} \big(504 p^{28} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{7} \\
&\quad - 3276 p^{24} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{6} + 8820 p^{20} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{5} \\
&\quad - 12474 p^{16} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{4} + 9520 p^{12} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{3} \\
&\quad - 3339 p^{8} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{2} \\
&\quad + 36 p^{4} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big) + 210\bigg)^{2}
\end{align}
$$



If the match is best of 5, the probability $$p_m^{(5)}:= P(\text{Player 1 wins Bo5})$$ is given by:

$$
p_m^{(5)} =
p_{s}^{3} \left(6 p_{s}^{2} - 15 p_{s} + 10\right)
$$


And in terms of $p$, we get the following expression:


$$
\begin{align}
p_m^{(5)} &= p^{72} \big(6 p^{48} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{12} \big(504 p^{28} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{7} \\
&\quad - 3276 p^{24} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{6} + 8820 p^{20} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{5} \\
&\quad - 12474 p^{16} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{4} + 9520 p^{12} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{3} \\
&\quad - 3339 p^{8} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{2} + 36 p^{4} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big) + 210\big)^{2} \\
&\quad - 15 p^{24} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{6} \big(504 p^{28} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{7} \\
&\quad - 3276 p^{24} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{6} + 8820 p^{20} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{5} \\
&\quad - 12474 p^{16} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{4} + 9520 p^{12} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{3} \\
&\quad - 3339 p^{8} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{2} + 36 p^{4} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big) + 210\big) \\
&\quad + 10\big) \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{18} \big(504 p^{28} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{7} \\
&\quad - 3276 p^{24} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{6} + 8820 p^{20} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{5} \\
&\quad - 12474 p^{16} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{4} + 9520 p^{12} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{3} \\
&\quad - 3339 p^{8} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big)^{2} + 36 p^{4} \big(- 20 p^{3} + 70 p^{2} - 84 p + 35\big) + 210\big)^{3}
\end{align}
$$

---

## Results and Interpretations

### Analytical solution

As shown above, we were able to derive tractable expressions for the probability of winning a game, set, and match as a function of the probability of winning a point, $p$.

$$p \rightarrow P_{\text{game}}(p) \rightarrow P_{\text{set}}(P_{\text{game}}(p)) \rightarrow P_{\text{match}}(P_{\text{set}}(\cdot))$$

These expressions compound on the previous one and that compounding causes the amplification effect. they also get more and more complex as we go up the hierarchy and they are difficult to interpret, even though they are just polynomials. For better interpretability, we plotted each of the three functions in the graph below for the case of a match played to the best of 5 sets.

![Tennis Probability Analysis](./data/tennis_probabilities_split.png)

Here, we can clearly see the amplification effect mentioned before, where small difference in the probability of winning a point compound heavily as we go up the hierarchy to the match level. At the match level, we get the largest amplification and hence the highest sensitivity to changes in the point win probability.


### Key Insights

1. **Small Differences Compound**: A 5% point advantage (55% vs 50%) translates to a +42.7% match advantage in best-of-5.

2. **Diminishing Returns**: The amplification effect is strongest near 50% and saturates as $p$ approaches extremes.

3. **Format Sensitivity**: Best-of-5 matches, currently only played at Grand Slams, amplify advantages more than best-of-3, favoring stronger players.

4. **Strategic Implications**: The mathematical structure explains why
players work on small, detailed adjustments on their serve, forehand, return, etc. as they lead toconsistent tactical advantages and translate to larger probabilities of tournament success.


---

## Extensions and Future Work

### Potential Enhancements & Extensions

1. **Detailed Tiebreak Modeling**: Replace simplified tiebreak with full first-to-7, win-by-2 model; and 10-point tiebreak for a deciding set.

2. **Service/Return Asymmetry**: Use different point win probabilities for service and return games.

3. **Momentum Effects**: Players have their moments during games where they play better or worse, so incorporating state-dependent point win probabilities may lead to better model.

4. **Surface-Specific Parameters**: Players always have a preferred surface where they perform best (even though a lot fo players are now all-court), so modeling different surfaces may lead to better predictions.

5. **Doubles matches**: It's easy to see how this modelling can be extended by introducing a specific win probability for each player.

6. **Numeric validation**: These Markov Chains can be simulated to validate our theoretical approach.

7. **Real data analysis**: A natural extension of this analysis is to compare it with real tennis data. We could estimate probabilities and see how these curves hold at scale across the tour.


---

## Conclusion

The mathematical analysis of tennis through absorbing Markov chains reveals the elegant structure underlying the sport's competitive dynamics. The hierarchical amplification effect—where small advantages compound through multiple levels—provides quantitative insight into why tennis exhibits clear skill hierarchies and why small improvements in fundamental abilities translate to disproportionate competitive advantages.


It is rewarding to bridge two of my passions (tennis and maths) to
derive a deeper understanding of one using tools from the other. It is indeed fascinating how even while making strong assumptions, we can extract key insights and principles that should hold at a macroscopic level.

Anecdotally, this study should also open the eyes of enthusiastic, amateur players that often believe they can return a serve or steal a game from a professional player, when in fact that is nearly impossible (or up to the pro's mercy).

As tennis continues to evolve with new technologies and playing styles, this mathematical foundation is general enough to provide a robust framework for understanding how changes in point-level capabilities propagate through the hierarchy to affect match outcomes.

---

## References

1. Pinsky, M. A., & Karlin, S. (2011). An introduction to stochastic modeling (4th ed.). Academic Press.

---
