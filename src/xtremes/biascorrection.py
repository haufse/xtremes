import numpy as np
import xtremes.topt as topt

def I_sb(j, bs):
    return np.arange(j,j+bs)

def D_n(n, bs):
    ## old:
    # return [(i, j) for i in np.arange(1,n-bs+2) 
    #         for j in np.arange(1,n-bs+2) 
    #         if np.intersect1d(I_sb(i, bs), I_sb(j, bs)).size == 0]
    
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

def hat_pi0(data, maxima=None, bs=None, stride='DBM'):
    if maxima is not None:
        bs = len(data) // len(maxima)
        print(bs)
    elif bs is not None:
        maxima = topt.extract_BM(data, bs, stride=stride)
    else:
        raise ValueError('Either maxima or block size must be provided')
    k = len(maxima)
    if stride == 'DBM':
        s = np.sum([exceedances(data, maxima, bs, i, j,stride='DBM')==1
                    for i in range(1,1+k)
                    for j in np.delete(np.arange(1,1+k), i-1)
                    ])
        return 4 * s/(k*(k-1))  
    if stride == 'SBM':
        n = len(data)
        s = np.sum([exceedances(data, maxima, bs, i, j,stride='SBM')==1 
                    for (i,j) in D_n(n, bs)
                    ])
        return 4 * s/len(D_n(n, bs))
    
from scipy.special import gamma, digamma
from scipy.optimize import root_scalar

def Upsilon(x, rho0):
    return rho0 * gamma(x+1) + (1-rho0) * gamma(x)
def Upsilon_derivative(x, rho0):
    return rho0 * gamma(x+1) * digamma(x+1) + (1-rho0) * gamma(x) * digamma(x)
def Pi(x, rho0):
    return 1/x - Upsilon_derivative(1+x, rho0)/Upsilon(1+x, rho0)+rho0/2 - np.euler_gamma # 0.5772156649015328606065120#np.euler_gamma
def Psi(a, a_true, rho0):
    frc = a/a_true
    term = 1/frc - Upsilon_derivative(1+frc, rho0)/Upsilon(1+frc, rho0)+rho0/2 - np.euler_gamma#0.5772156649015328606065120#np.euler_gamma
    return 2/a_true * term

def a1_asy(a_true, rho0):
    sol = root_scalar(Psi, args=(a_true, rho0), bracket=[1e-2, 100])
    return sol.root

def varpi(rho0):
    sol = root_scalar(Pi, args=(rho0,), bracket=[0.01, 10])
    return sol.root

def z0(rho0):
    return 1/2*Upsilon(1+varpi(rho0), rho0) 