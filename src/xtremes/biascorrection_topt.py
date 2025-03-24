
from scipy.special import gamma, digamma
from scipy.optimize import root_scalar
from tqdm import tqdm
import numpy as np
from xtremes import topt

def I_sb(j, bs):
    return np.arange(j,j+bs)
def D_n(n, bs):
    return [(i, j) for i in np.arange(1,n-bs+2) 
            for j in np.arange(1,n-bs+2) 
            if np.intersect1d(I_sb(i, bs), I_sb(j, bs)).size == 0]

def exceedances(data, maxima, bs, i, j, stride='DBM'):
    r"""Caluclate number of exceedances of the i-th maximum in the j-th block"""
    if stride == 'DBM':
        return np.sum(data[(j-1)*bs:j*bs] > maxima[i-1])
    if stride == 'SBM':
        return np.sum(data[j:j+bs] > maxima[i-1])

def pbar(m, data, maxima=None, bs=None, stride='DBM'):
    if maxima is not None:
        bs = len(data) // len(maxima)
        print(bs)
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
                    for (i,j) in D_n(n, bs)
                    ])
        return  s/len(D_n(n, bs))

def hat_pi(t, data, maxima=None, bs=None, stride='DBM'):
    if t == 1:
        return 4*pbar(t, data, maxima, bs, stride)
    else:
        return 4*pbar(t, data, maxima, bs, stride) - 2 * np.sum([hat_pi(t-k, data, maxima, bs, stride)*pbar(k, data, maxima, bs, stride) for k in range(1,t)])
    
def hat_pis(t, data, maxima=None, bs=None, stride='DBM'):
    hp = [hat_pi(k, data, maxima, bs, stride) for k in range(1,t+1)]
    return np.append([0],np.array(hp))

def convolve_probabilities(pi, l, i):
    """
    Compute P(sum_{h=1}^l K_h = i) using iterative convolutions of pi.
    
    Parameters:
    - pi: np.array of probabilities [pi(1), pi(2), ..., pi(r)]
    - l: Number of random variables in the sum
    - i: Desired sum value
    
    Returns:
    - P(sum_{h=1}^l K_h = i)
    """
    if l == 0:
        return 0
    if l == 1:
        # Base case: first convolution is just pi
        return pi[i] if i <= len(pi) else 0.0

    # Start with the first distribution
    result = pi.copy()

    # Perform (l - 1) convolutions
    for _ in range(l-1):
        result = np.convolve(result, pi)

    # Retrieve the desired probability
    return result[i-1] if i <= len(result) else 0.0


def hat_frakp(pi, j, l):
    if l == 0:
        return 1
    elif j > l:
        return (-1)**l/gamma(l+1)*np.sum([convolve_probabilities(pi, l, i) for i in range(l, j)])
    else:
        return 0

def ups(x, t, pi):
    if t==1:
        return hat_frakp(pi, t, 0)*gamma(1+x)
    return hat_frakp(pi, t, 0)*gamma(1+x) + np.sum([hat_frakp(pi, t, l)*(-1)**l *x*gamma(l+x) for l in range(1,t)])

def ups_d(x, t, pi):
    if t==1:
        return hat_frakp(pi, t, 0)*gamma(1+x)*digamma(x+1)
    return hat_frakp(pi, t, 0)*gamma(1+x)*digamma(x+1) + np.sum([hat_frakp(pi, t, l)*(-1)**l *gamma(l+x) * (x*digamma(x+l)+1) for l in range(1,t)])

def Pi(x, t, pi):
    return 1/x - ups_d(x, t, pi)/ups(x, t, pi) -np.euler_gamma+1/t*np.sum([np.sum([(-1)**l*hat_frakp(pi, j, l)*gamma(l) for l in range(1,j+1)]) for j in range(1,t+1)])

def varpi(t, pi):
    sol = root_scalar(Pi, args=(t,pi,), bracket=[0.01, 10])
    return sol.root
