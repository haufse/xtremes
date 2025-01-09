## Implements the bias correction based on Bücher and Jennessen
# only for the Frechet case relevant

import numpy as np
from scipy.special import gamma as sps_gamma
from scipy.special import digamma
from scipy.optimize import root_scalar
from tqdm import tqdm
import xtremes.topt as topt

def I_sb(j, bs):
    return np.arange(j,j+bs)
def D_n(n, bs):
    return [(i, j) for i in np.arange(1,n-bs+2) 
            for j in np.arange(1,n-bs+2) 
            if np.intersect1d(I_sb(i, bs), I_sb(j, bs)).size == 0]
# does the same in fast
def D_n_fast(n, bs):
    idx = np.arange(1, n-bs+2)
    # You can create meshgrids or just do two loops:
    pairs = []
    for i in idx:
        # All j to the left: j+bs <= i
        left_js = idx[idx + bs <= i]
        # All j to the right: j >= i+bs
        right_js = idx[idx >= i + bs]
        for j in left_js:
            pairs.append((i, j))
        for j in right_js:
            pairs.append((i, j))
    return pairs

def exceedances(data, maxima, bs, i, j, stride='DBM'):
    r"""Caluclate number of exceedances of the i-th maximum in the j-th block"""
    if stride == 'DBM':
        return np.sum(data[(j-1)*bs:j*bs] > maxima[i-1])
    if stride == 'SBM':
        return np.sum(data[j:j+bs] > maxima[i-1])

def pbar(m, data, maxima=None, bs=None, stride='DBM'):
    if maxima is not None:
        bs = len(data) // len(maxima)
        # print(bs)
    elif bs is not None:
        maxima = topt.extract_BM(data, bs, stride=stride)
    else:
        raise ValueError('Either maxima or block size must be provided')
    k = len(maxima)
    if stride == 'DBM':
        s = np.sum([exceedances(data, maxima, bs, i, j,stride='DBM')==m
                    for i in range(1,1+k)
                    for j in np.delete(np.arange(1,1+k), i-1)
                    ])
        return  s/(k*(k-1))  
    if stride == 'SBM':
        n = len(data)
        s = np.sum([exceedances(data, maxima, bs, i, j,stride='SBM')==m
                    for (i,j) in D_n_fast(n, bs)
                    ])
        return  s/len(D_n_fast(n, bs))

# does the same in fast
def pbar_dbm_fast(m, data, maxima=None, bs=None):
    """
    A faster version of the DBM part using precomputed cumulative sums.
    Returns pbar(m) for DBM stride.
    """
    # Either maxima is given, or we compute it from data & bs
    if maxima is None:
        if bs is None:
            raise ValueError('Either maxima or block size must be provided')
        else:
            # You presumably have some function topt.extract_BM(...)
            maxima = topt.extract_BM(data, bs, stride='DBM')
    else:
        # If maxima given, deduce bs from data
        bs = len(data) // len(maxima)
    
    k = len(maxima)
    n = len(data)
    
    # 1) Build X: shape (k, n), X[i, t] = 1 if data[t] > maxima[i]
    #    This is a vectorized comparison across i,t
    #    We'll do data in one dimension => shape is (1,n) vs (k,1):
    #    then broadcast the comparison
    # 
    #    But watch out for shape mismatch: we want 
    #    X[i,t] = data[t] > maxima[i].
    #    We can do:
    X = (data[None, :] > maxima[:, None]).astype(int)
    
    # 2) Build Y: shape (k, n+1), cumulative sums of each row in X
    Y = np.zeros((k, n+1), dtype=int)
    np.cumsum(X, axis=1, out=Y[:, 1:])  # store cumsum from col1 onward
    
    # 3) For each i (0-based) and each j != i, the sum of the j-th block is:
    #    sum_{t=j*bs}^{(j+1)*bs - 1} X[i,t]
    #    = Y[i, (j+1)*bs] - Y[i, j*bs].
    #
    #    We can compute these block sums for each row i all at once:
    #    block_sums[i, j] = Y[i, (j+1)*bs] - Y[i, j*bs].
    
    # Make an array of block boundaries from 0, bs, 2bs, ... k*bs
    block_edges = np.arange(0, bs*(k+1), bs)
    
    # block_sums will have shape (k, k) because there are k blocks total
    # block_sums[i, j] = sum_{t in block j} X[i,t]
    block_sums = Y[:, block_edges[1:]] - Y[:, block_edges[:-1]]
    
    # 4) Now we want to count how many times block_sums[i,j] == m for j != i.
    #    Then we divide by k*(k-1).
    
    # Let's do it in a vectorized manner:
    mask_offdiag = np.ones((k, k), dtype=bool)
    np.fill_diagonal(mask_offdiag, False)  # j != i
    
    count = np.sum(block_sums[mask_offdiag] == m)
    
    return count / (k * (k - 1))

 
# def hat_pi(m, data, maxima=None, bs=None, stride='DBM'):
#     if m == 1:
#         return 4*pbar(m, data, maxima, bs, stride)
#     else:
#         return 4*pbar(m, data, maxima, bs, stride) - 2 * np.sum([hat_pi(m-k, data, maxima, bs, stride)*pbar(k, data, maxima, bs, stride) for k in range(1,m)])
 
# def hat_pis(m, data, maxima=None, bs=None, stride='DBM'):
#     hp = [hat_pi(k, data, maxima, bs, stride) for k in range(1,m+1)]
#     return np.array(hp)

def hat_pis(m, data, maxima=None, bs=None, stride='DBM', pbars=None):
    if pbars == None:
        #print('\r Computing pbars',end='')
        #pbars = [pbar(k, data, maxima, bs, stride) for k in range(1,m+1)]
        if stride=='DBM':
            pbars = [pbar_dbm_fast(k, data, maxima, bs) for k in range(1,m+1)]
        #print('\r pbars computed  ',end='')

    hps = [4*pbars[0]]
    if m == 1:
        return hps
    else:
        for q in tqdm(range(1,m)):
            hps.append(4*pbars[q]-2*np.sum([hps[-k]*pbars[k] for k in range(1,q)]))
    return hps
    


def convolve_probabilities(pi, l, k):
    """
    Compute P(sum_{i=1}^l K_i = k) using iterative convolutions of pi.
    
    Parameters:
    - pi: np.array of probabilities [pi(1), pi(2), ..., pi(r)]
    - l: Number of random variables in the sum
    - k: Desired sum value
    
    Returns:
    - P(sum_{i=1}^l K_i = k)
    """
    if l == 1:
        # Base case: first convolution is just pi
        return pi[k] if k <= len(pi) else 0.0

    # Start with the first distribution
    result = pi.copy()

    # Perform (l - 1) convolutions
    for _ in range(1, l):
        result = np.convolve(result, pi)

    # Retrieve the desired probability
    return result[k] if k - 1 < len(result) else 0.0

def hat_frakp(pi, r, l):
    return (-1)**l/sps_gamma(l+1)*np.sum([convolve_probabilities(pi, l, k) for k in range(l, r)])

def ups(x, r, pi):
    return np.sum([hat_frakp(pi, r, l)*(-1)**l *x*sps_gamma(l+x) if x+l!=0 else hat_frakp(pi, r, l)*(-1)**l  for l in range(r)])

def ups_d(x, r, pi):
    return np.sum([hat_frakp(pi, r, l)*(-1)**l *sps_gamma(l+x) * (x*digamma(x+l)+1) if x+l!=0 else hat_frakp(pi, r, l)*(-1)**l *(-np.euler_gamma) for l in range(r)])

def Pi(x,r,pi):
    return 1/x - ups_d(x, r, pi)/ups(x, r, pi) +1/r*np.sum([ups_d(0,j,pi) for j in range(1,r+1)])

# def varpi(r,pi):
#     sol = root_scalar(Pi, args=(r,pi,), bracket=[0.001, 100])
#     return sol.root
def varpi(r, pi):
    """
    Attempt to find x in [0.001, 100] s.t. Pi(x, r, pi) = 0.
    If that fails, return the x that minimizes |Pi(x, r, pi)|.
    """
    # Wrap your function Pi in a single-argument function for root_scalar.
    def objective(x):
        return Pi(x, r, pi)

    # First, try root finding:
    try:
        sol = root_scalar(objective, bracket=[0.001, 20])
        # If it converged, great!
        if sol.converged:
            return sol.root
    except Exception:
        pass
    return 100