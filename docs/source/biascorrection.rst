===============================
Bias Correction Tutorial
===============================

In this tutorial, we will explore how to work with the `topt` module from `xtremes` to apply bias correction techniques in extreme value analysis. Specifically, we will cover:

- Generating index ranges for sliding and disjoint block methods (`I_sb` and `D_n`).
- Computing exceedances using `exceedances`.
- Estimating cluster size probability using `hat_pi0`.
- Utilizing the `Upsilon`, `Pi`, and `Psi` functions for bias correction.
- Solving for bias-corrected parameters with `varpi` and `a1_asy`.

Each section includes code examples for clarity.

Step 1: Generating Block Indices
================================
In time series analysis, blocks of data are used to analyze cluster sizes and exceedances. We define two key functions:

### `I_sb(j, bs)`: Generate indices for a sliding block
```python
import numpy as np

def I_sb(j, bs):
    return np.arange(j, j + bs)

# Example usage
print(I_sb(3, 5))  # Output: [3, 4, 5, 6, 7]
```

### `D_n(n, bs)`: Generate index pairs for disjoint blocks
```python
def D_n(n, bs):
    idx = np.arange(1, n - bs + 2)
    pairs = []
    for i in idx:
        left_js = idx[idx + bs <= i]
        right_js = idx[idx >= i + bs]
        for j in left_js:
            pairs.append((i, j))
        for j in right_js:
            pairs.append((i, j))
    return pairs

# Example usage
print(D_n(5, 2))  # Output: [(1, 3), (1, 4), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]
```

Step 2: Computing Exceedances
==============================
The `exceedances` function counts values exceeding a given threshold in a block.
```python
def exceedances(data, maxima, bs, i, j, stride='DBM'):
    if stride == 'DBM':
        return np.sum(data[(j-1)*bs:j*bs] > maxima[i-1])
    if stride == 'SBM':
        return np.sum(data[j:j+bs] > maxima[i-1])

# Example usage
data = np.array([1, 3, 5, 2, 6, 4, 7, 9])
maxima = np.array([5, 7])
print(exceedances(data, maxima, bs=2, i=1, j=3, stride='DBM'))  # Output: 1
```

Step 3: Estimating Cluster Size Probability
===========================================
The function `hat_pi0` estimates the probability of cluster size 1.
```python
import xtremes.topt as topt

def hat_pi0(data, maxima=None, bs=None, stride='DBM'):
    if maxima is not None:
        bs = len(data) // len(maxima)
    elif bs is not None:
        maxima = topt.extract_BM(data, bs, stride=stride)
    else:
        raise ValueError('Either maxima or block size must be provided')
    k = len(maxima)
    if stride == 'DBM':
        s = np.sum([exceedances(data, maxima, bs, i, j, stride='DBM') == 1
                    for i in range(1, 1 + k)
                    for j in np.delete(np.arange(1, 1 + k), i-1)])
        return 4 * s / (k * (k-1))

# Example usage
print(hat_pi0(data, maxima, bs=2, stride='DBM'))
```

Step 4: Using Bias Correction Functions
=======================================
We define functions for the `Upsilon`, `Pi`, and `Psi` functions used in bias correction.

```python
from scipy.special import gamma, digamma, polygamma
from scipy.optimize import root_scalar

def Upsilon(x, rho0):
    return rho0 * gamma(x+2) + (1-rho0) * gamma(x+1)

def Upsilon_derivative(x, rho0):
    return rho0 * gamma(x+2) * digamma(x+2) + (1-rho0) * gamma(x+1) * digamma(x+1)

def Pi(x, rho0):
    return 1/x - Upsilon_derivative(x, rho0)/Upsilon(x, rho0) + rho0/2 - np.euler_gamma

def Psi(a, a_true, rho0):
    vp = a / a_true
    term = 1 / vp - Upsilon_derivative(vp, rho0) / Upsilon(vp, rho0) + rho0/2 - np.euler_gamma
    return 2 / a_true * term
```

Step 5: Solving for Bias-Corrected Parameters
=============================================
To obtain bias-corrected parameters, we use numerical root-finding methods.

```python
def varpi(rho0):
    sol = root_scalar(Pi, args=(rho0,), bracket=[0.01, 10])
    return sol.root

def a1_asy(a_true, rho0):
    sol = root_scalar(Psi, args=(a_true, rho0), bracket=[1e-2, 100])
    return sol.root

# Example usage
rho0 = 0.5
print(varpi(rho0))  # Computes bias-corrected root
```

Conclusion
==========
In this tutorial, we explored:
- How to generate block indices.
- Compute exceedances within blocks.
- Estimate cluster size probability using `hat_pi0`.
- Use the `Upsilon`, `Pi`, and `Psi` functions for bias correction.
- Solve for bias-corrected parameters using numerical root-finding.

These methods are essential for extreme value analysis in time series data and improving MLE estimates by accounting for bias.

