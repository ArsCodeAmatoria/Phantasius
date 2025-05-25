# LaTeX Fix Demonstration

## Problem: Missing opening declaration

**This is BROKEN:**
```
A simple matrix:

a & b \\ c & d \end{pmatrix}$$
```

**This is CORRECT:**
```
A simple matrix:

$$\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}$$
```

## The Complete Corrected Version

A simple matrix:

$$\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}$$

The identity matrix:

$$I = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}$$

A vector in quantum mechanics:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

Where $|\alpha|^2 + |\beta|^2 = 1$. 