---
title: "Testing KaTeX Math Expressions"
date: "2025-05-29"
excerpt: "A comprehensive test post to verify that mathematical expressions render correctly with KaTeX"
tags: ["test", "math", "katex"]
---

# Testing KaTeX Math Expressions

Let's test both inline and display math expressions to ensure proper rendering.

## Inline Math

The Pythagorean theorem states that $a^2 + b^2 = c^2$ for right triangles.

The golden ratio is $\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618$.

In physics, Einstein's mass-energy equivalence is expressed as $E = mc^2$.

The quadratic formula can be written inline as $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.

## Display Math

The quadratic formula:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

Euler's identity (often called the most beautiful equation):

$$e^{i\pi} + 1 = 0$$

The Schrödinger equation:

$$i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)$$

The wave equation:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$$

## Greek Letters and Symbols

Some Greek letters commonly used in mathematics:

- Alpha: $\alpha$ and $\Alpha$
- Beta: $\beta$ and $\Beta$
- Gamma: $\gamma$ and $\Gamma$
- Delta: $\delta$ and $\Delta$
- Epsilon: $\epsilon$ and $\varepsilon$
- Phi: $\phi$ and $\Phi$
- Lambda: $\lambda$ and $\Lambda$
- Mu: $\mu$ and $M$
- Pi: $\pi$ and $\Pi$
- Sigma: $\sigma$ and $\Sigma$
- Omega: $\omega$ and $\Omega$

Special symbols:
- Infinity: $\infty$
- Partial derivative: $\partial$
- Integral: $\int$
- Sum: $\sum$
- Product: $\prod$

## Complex Expressions

The MOND acceleration scale:

$$a_0 = 1.2 \times 10^{-10} \text{ m/s}^2$$

The holographic principle entropy bound:

$$S \leq \frac{A}{4G\hbar}$$

Where $A$ is the surface area, $G$ is Newton's gravitational constant, and $\hbar$ is the reduced Planck constant.

Maxwell's equations:

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$$

$$\nabla \cdot \mathbf{B} = 0$$

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

$$\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}$$

## Matrix and Vector Notation

### Basic 2×2 Matrix

A simple 2×2 matrix representation:

$$\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}$$

This matrix has elements arranged in 2 rows and 2 columns.

### Identity Matrix

Next, we show the 3×3 identity matrix:

$$I = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}$$

The identity matrix has 1s on the diagonal and 0s elsewhere.

### Variable Matrix

Here's a matrix with variable elements:

$$A = \begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{pmatrix}$$

Each element is labeled with its row and column indices.

### Numerical Example

Example matrix $A$ with specific values:

$$A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}$$

This is a concrete example of a 2×2 matrix.

### Column Vector

Column vector $\mathbf{x}$ representation:

$$\mathbf{x} = \begin{pmatrix}
x \\
y
\end{pmatrix}$$

Vectors are special cases of matrices with only one column.

### Matrix-Vector Multiplication

The result of multiplying matrix $A$ by vector $\mathbf{x}$:

$$A\mathbf{x} = \begin{pmatrix}
x + 2y \\
3x + 4y
\end{pmatrix}$$

This demonstrates matrix-vector multiplication in action.

## Quantum Mechanics Notation

A vector in quantum mechanics:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

Where $|\alpha|^2 + |\beta|^2 = 1$.

The expectation value:

$$\langle A \rangle = \langle\psi|A|\psi\rangle$$

The commutation relation:

$$[X, P] = i\hbar$$

A quantum state superposition:

$$|\psi\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + e^{i\phi}|1\rangle\right)$$

## Calculus and Analysis

Definite integral:

$$\int_a^b f(x) \, dx$$

Multiple integral:

$$\iint_D f(x,y) \, dx \, dy$$

Partial derivatives:

$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial y}\right)$$

Taylor series:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

## Set Theory and Logic

Set operations:

$$A \cup B = \{x : x \in A \text{ or } x \in B\}$$

$$A \cap B = \{x : x \in A \text{ and } x \in B\}$$

Logical quantifiers:

$$\forall x \in \mathbb{R}, \exists y \in \mathbb{R} : y = x^2$$

## Number Theory

The prime counting function:

$$\pi(x) \sim \frac{x}{\ln x}$$

Euler's totient function:

$$\phi(n) = n \prod_{p|n} \left(1 - \frac{1}{p}\right)$$

## Conclusion

This page tests various LaTeX mathematical expressions to ensure proper rendering with KaTeX. All expressions should display correctly without parsing errors. 