# AI-H - 2026Spring - Practice-VIII - Supervised Learning: Movie Profit Prediction Model Comparison

**Name:** Yuxuan HOU

**Student ID:** 12413104

**Date: **2026.04.22

## Dataset Overview

| Movie Name   | A Score ($f_1$) | B Score ($f_2$) | Profit? | Label $y$ |
| ------------ | --------------- | --------------- | ------- | --------- |
| Pellet Power | 1               | 1               | No      | $-1$      |
| Ghosts!      | 3               | 2               | Yes     | $+1$      |
| Pac is Bac   | 4               | 5               | No      | $-1$      |
| Not a Pizza  | 3               | 4               | Yes     | $+1$      |
| Endless Maze | 2               | 3               | Yes     | $+1$      |

------

## Task 1: Data Visualization and Linear Separability Analysis

### 1.1 Scatter Plot (conceptual sketch)

```
B (Critic B)
 5 |           -          (Pac is Bac)
 4 |     +                (Not a Pizza)
 3 |  +                   (Endless Maze)
 2 |     +                (Ghosts!)
 1 |  -                   (Pellet Power)
   +------------------>  A (Critic A)
      1  2  3  4  5
```

- **"+"** denotes profitable (Yes): $(3,2),(3,4),(2,3)$
- **"−"** denotes non-profitable (No): $(1,1),(4,5)$

### 1.2 Linear Separability Proof

Observe that the two negative points $(1,1)$ and $(4,5)$ lie on the line

$$ 4x - 3y = 1. $$

Parameterize the segment between them as $(1+3t,,1+4t),, t \in [0,1]$. Setting $x = 3$ gives $t = 2/3$, yielding the point $(3, 11/3) \approx (3, 3.67)$, which lies **inside** the convex hull of the positive points (the triangle with vertices $(3,2),(3,4),(2,3)$).

Because the **convex hulls of the two classes intersect**, by the Separating Hyperplane Theorem, **no linear boundary can separate them**. The dataset is **not linearly separable**.

Intuitively, both low scores $(1,1)$ and high scores $(4,5)$ indicate failure, while medium scores predict success — this resembles an **XOR-like** pattern.

### 1.3 Which Model is More Suitable?

| Model                   | Analysis                                                     | Verdict            |
| ----------------------- | ------------------------------------------------------------ | ------------------ |
| **Linear SVM**          | Cannot handle non-linear separability without kernels. Soft margin helps but cannot achieve perfect accuracy linearly. | ✗ (without kernel) |
| **Neural Network (NN)** | Can learn arbitrary non-linear boundaries, but with only 5 samples it is prone to severe overfitting. | △                  |
| **Decision Tree (DT)**  | Splits axis-aligned; naturally captures "medium scores → Yes, extreme scores → No" with few splits. Interpretable. | ✓                  |

**Recommendation:** A **Decision Tree** is most suitable given the small dataset size and the non-linear, axis-aligned nature of the decision boundary. An SVM with an RBF kernel is a close second.

------

## Task 2: Perceptron Model Training and Feature Selection

### 2.1 Feature Representation

Each sample is represented as

$$ \mathbf{x}_i = (f_0,, f_1,, f_2)^\top = (1,, A_i,, B_i)^\top, \quad y_i \in {-1, +1}. $$

### 2.2 Weight Update Rule

Perceptron prediction: $\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x})$.
 If $\hat{y}_i \neq y_i$, update:

$$ \mathbf{w} \leftarrow \mathbf{w} + y_i , \mathbf{x}_i. $$

If correctly classified, leave $\mathbf{w}$ unchanged.

### 2.3 First Update

Initialize $\mathbf{w}^{(0)} = (0, 0, 0)^\top$.

Take the first training example — **Pellet Power**: $\mathbf{x}_1 = (1, 1, 1)^\top,; y_1 = -1$.

$$ \mathbf{w}^{(0)\top} \mathbf{x}_1 = 0 \cdot 1 + 0 \cdot 1 + 0 \cdot 1 = 0. $$

$\operatorname{sign}(0)$ is conventionally treated as $+1$, which disagrees with $y_1 = -1$. Thus we update:

$$ \mathbf{w}^{(1)} = \mathbf{w}^{(0)} + y_1, \mathbf{x}_1 = (0,0,0)^\top + (-1)(1,1,1)^\top = (-1, -1, -1)^\top. $$

$$ \boxed{\mathbf{w}^{(1)} = (-1,,-1,,-1)^\top} $$

### 2.4 Can a Perceptron Perfectly Classify These Rules?

#### (i) $A + B > 8 \Rightarrow$ Success

Set $w_0 = -8,, w_1 = 1,, w_2 = 1$. Then

$$ \mathbf{w}^\top \mathbf{x} = -8 + A + B > 0 \iff A + B > 8. $$

This is a single linear half-plane.
 **✓ Yes — a perceptron can represent this perfectly.**

#### (ii) Success iff each critic gives a score of 2 or 3

Positive region = ${(A,B) : A \in {2,3} \text{ and } B \in {2,3}}$.

Consider the horizontal line $B = 2$: points $(1,2)$ (−), $(2,2)$ (+), $(4,2)$ (−). A positive point lies strictly between two negatives on the same line — impossible to separate with a linear boundary in the original feature space.

**✗ No — the region is a bounded rectangle, which cannot be expressed as a single linear half-space.** It would require at least the intersection of 4 linear inequalities (i.e., multiple perceptrons or a non-linear model).

#### (iii) Success iff both critics agree ($A = B$)

The positive set is the line $A = B$ itself (a measure-zero set); everywhere else is negative. A single linear decision boundary $\mathbf{w}^\top \mathbf{x} = 0$ divides the plane into **two** half-spaces, but both "above" and "below" the diagonal contain negatives.

**✗ No — a perceptron cannot realize this XOR-like pattern with only the raw features $(A,B)$.**

> **Remark:** Scenarios (ii) and (iii) become perceptron-learnable after a non-linear feature expansion (e.g., polynomial kernels or additional hand-crafted features like $|A-B|$).

------

## Task 3: Soft Margin SVM Analysis

### 3.1 Formulation

For non-separable data, the **soft margin SVM** introduces slack variables $\xi_i \geq 0$:

$$ \min_{\mathbf{w},,b,,\boldsymbol{\xi}} \quad \frac{1}{2}|\mathbf{w}|^2 + C \sum_{i=1}^{n} \xi_i $$

$$ \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i. $$

- $\xi_i = 0$: sample is correctly classified outside the margin.
- $0 < \xi_i \leq 1$: sample is inside the margin but correctly classified.
- $\xi_i > 1$: sample is **misclassified**.

### 3.2 Role of the Regularization Parameter $C$

| Regime                               | Effect                                                       |
| ------------------------------------ | ------------------------------------------------------------ |
| **Large $C$** (e.g., $C \to \infty$) | Heavy penalty on $\xi_i$ ⇒ few training errors allowed ⇒ narrower margin, decision boundary closely fits training data ⇒ **risk of overfitting**. |
| **Small $C$** (e.g., $C \to 0$)      | Low penalty on $\xi_i$ ⇒ more violations tolerated ⇒ wider margin, smoother boundary ⇒ **risk of underfitting**. |

Thus $C$ is the dial that controls the **bias–variance trade-off**.

### 3.3 Trade-off Between Margin and Error

The objective has two competing terms:

$$ \underbrace{\tfrac{1}{2}|\mathbf{w}|^2}*{\text{inverse of margin width}} ;+; \underbrace{C \sum_i \xi_i}*{\text{total classification error}} $$

- Maximizing the margin ($|\mathbf{w}|^2$ small) improves generalization (VC-theoretically).
- Minimizing training error ($\sum \xi_i$ small) improves fit.

$C$ sets the relative weight. In our 5-point dataset — which is not linearly separable — a **moderate $C$** (selected via cross-validation) would yield a reasonable margin while permitting the two negative "outliers" $(1,1)$ and $(4,5)$ to violate the margin, at the cost of at least one training error. Using a **non-linear kernel** (e.g., RBF) would allow soft-margin SVM to classify all five points correctly.

------

## Task 4: Decision Tree Model Construction

### 4.1 Splitting Criterion — Information Gain

Total entropy of the root (3 Yes, 2 No):

$$ H(S) = -\frac{3}{5}\log_2\frac{3}{5} - \frac{2}{5}\log_2\frac{2}{5} \approx 0.971 \text{ bits}. $$

We evaluate candidate splits:

| Split      | Left subset | Right subset | Weighted Entropy                            | Info Gain |
| ---------- | ----------- | ------------ | ------------------------------------------- | --------- |
| $A \leq 1$ | ${(1,1,-)}$ | 3Y, 1N       | $\tfrac{1}{5}(0)+\tfrac{4}{5}(0.811)=0.649$ | **0.322** |
| $A \leq 2$ | 1Y, 1N      | 2Y, 1N       | $0.951$                                     | $0.020$   |
| $A \leq 3$ | 3Y, 1N      | ${(4,5,-)}$  | $0.649$                                     | **0.322** |
| $B \leq 1$ | ${(1,1,-)}$ | 3Y, 1N       | $0.649$                                     | **0.322** |
| $B \leq 3$ | 2Y, 1N      | 1Y, 1N       | $0.951$                                     | $0.020$   |
| $B \leq 4$ | 3Y, 1N      | ${(4,5,-)}$  | $0.649$                                     | **0.322** |

Tied best splits with gain $0.322$. Choose **$A \leq 1$** (others are equivalent).

### 4.2 Tree Construction

**Root:** split on $A \leq 1$.

- **Left** ($A \leq 1$): one sample, Pellet Power $\Rightarrow$ **No** ✓
- **Right** ($A > 1$): ${(3,2,+),(4,5,-),(3,4,+),(2,3,+)}$, entropy $=0.811$.
  - Split on $A \leq 3$: Left = all 3 positives (pure), Right = $(4,5,-)$ (pure). Info gain $= 0.811$.

**Final tree:**

```
                 [A ≤ 1?]
                /       \
              Yes        No
              /           \
          No (-)         [A ≤ 3?]
                          /    \
                        Yes     No
                        /        \
                   Yes (+)     No (-)
```

In words:

$$ \hat{y} = \begin{cases} \text{No},  & A = 1 \ \text{Yes}, & 2 \leq A \leq 3 \ \text{No},  & A \geq 4 \end{cases} $$

This depth-2 tree achieves **100% training accuracy**.

### 4.3 Depth and Pruning

- **Depth too shallow** $\Rightarrow$ underfits (cannot capture medium-vs-extreme pattern).
- **Depth too deep** $\Rightarrow$ each leaf holds very few samples (with only 5 points, a depth-3 tree would trivially memorize noise) $\Rightarrow$ **overfitting**.
- **Pruning** combats overfitting:
  - *Pre-pruning*: stop splitting when info gain < threshold, leaf size < $n_{\min}$, or max depth reached.
  - *Post-pruning* (e.g., cost-complexity pruning): grow a full tree, then collapse subtrees whose removal does not significantly increase validation error, minimizing $$ R_\alpha(T) = R(T) + \alpha,|T_{\text{leaves}}|. $$

For our tiny dataset, a depth limit of 2 (as derived) is already an implicit form of pruning and is appropriate.

------

## Task 5: Model Comparison and Selection

### 5.1 Side-by-Side Comparison

| Aspect                     | **SVM (soft margin / RBF)**          | **Neural Network**                           | **Decision Tree**              |
| -------------------------- | ------------------------------------ | -------------------------------------------- | ------------------------------ |
| Handles non-linearity      | ✓ via kernel                         | ✓ via hidden layers                          | ✓ axis-aligned splits          |
| Performance on 5 samples   | Reasonable, but kernel tuning needed | Poor — severe overfitting with so few points | Excellent — splits are natural |
| Interpretability           | Low (especially with kernels)        | Very low (black box)                         | **Very high** (if-else rules)  |
| Training cost              | Moderate ($O(n^2)$–$O(n^3)$)         | High; many hyperparameters                   | Low                            |
| Sensitivity to hyperparams | High ($C$, kernel, $\gamma$)         | Very high (architecture, lr, …)              | Low (depth, min-samples)       |
| Robustness to outliers     | Soft margin tolerates them           | Can memorize them                            | Isolates them in leaves        |
| Data requirement           | Small–medium                         | **Large**                                    | Small                          |

### 5.2 Comprehensive Evaluation

For this **specific problem** (5 samples, 2 features, non-linear XOR-like structure, need for interpretability to justify greenlighting a film):

- A **Neural Network** is inappropriate — only 5 training examples cannot support reliable weight estimation; it would either underfit drastically or memorize the set.
- A **Kernel SVM** (RBF) would work in practice, but requires tuning $C$ and $\gamma$ via cross-validation, which is statistically unstable with 5 points, and the model is essentially a black box to a film executive.
- A **Decision Tree** is the clear winner: it achieves perfect training accuracy with a 2-level depth, produces **human-readable rules** ("the film succeeds iff at least one critic gives a middling score of 2 or 3"), and naturally captures the non-linear pattern without any kernel trick.

### 5.3 Final Recommendation

$$ \boxed{\text{Choose the Decision Tree.}} $$

**Reasons:**

1. **Accuracy:** perfectly fits the non-linearly-separable training data with minimal depth.
2. **Interpretability:** decision rules can be communicated directly to producers and writers.
3. **Data efficiency:** works well even in small-sample regimes where NN/SVM may be unstable.
4. **Low computational cost** and **few hyperparameters**, making it easy to tune and deploy.

If the dataset were expanded to hundreds or thousands of movies, an **ensemble method** such as **Random Forest** or **Gradient Boosted Trees** would be the preferred upgrade, combining the interpretability of trees with lower variance. A kernel SVM or a small NN with regularization would become competitive alternatives in that regime.