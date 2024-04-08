"""
test_xtremes.miscellaneous - Python sublibrary for diverse functions.

.. code-block:: python

    # Import test_xtremes
    import test_xtremes.miscellaneous as misc

    # Call its only function
    misc.sigmoid(1)
"""

import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import genextreme, invweibull, weibull_max, gumbel_r, genpareto, cauchy
from scipy.special import gamma as Gamma
from pynverse import inversefunc
import pickle
import warnings


def sigmoid(x):
    """
    Returns the sigmoid of a function.

    :param x: input
    :type x: int, float list or np.array
    :return: The sigmoid of the input
    :rtype: np.ndarray[float]
    """
    x = np.array(x)
    return 1/(1+np.exp(-x))

def invsigmoid(y):
    if y < 0 or y > 1:
        return ValueError('Sigmoid only maps to (0, 1)!')
    elif y < np.exp(-30):
        return -30 # exp(-30)=0
    elif y > 1-np.exp(30):
        return 30 # exp(30)=infty
    else:
        return inversefunc(sigmoid, y_values=y).item()

# the GEV
def gev(x, gamma=0, mu=0, sigma=1):
    y = (x-mu)/sigma #standard
    if gamma == 0:
        return np.exp(-np.exp(-y))*np.exp(-y)/sigma
    elif 1+gamma*y>0:
        return np.exp(-(1+gamma*y)**(-1/gamma))*(1+gamma*y)**(-1/gamma-1)/sigma
    else:
        return 0
# the GEV for lists
def GEV(x, gamma=0, mu=0, sigma=1):
    g = lambda x: gev(x, gamma, mu, sigma)
    return list(map(g, x))

# log-likelihood
def ll_gev(x, gamma=0, mu=0, sigma=1, pi=1, option=1, max_only=False, second_only=False):
    """
    Computes the log likelihood of the GEV
    option: int
        if ==1: only use maximum for likelihood
        if ==2: use maximum + 2. largest for likelihood
        if ==3: use maximum + 2. largest + additional param for likelihood
        max only and second only for splitting up option 3
    """
    # numerically more stable?
    sigma = np.abs(sigma)
    #if sigma < 0.1:
    #    print('sigma close to 0!')
    #    sigma += 0.1
    y = (x-mu)/sigma #standard
    if option == 1:
        if np.abs(gamma) < 0.01:
            return -np.log(sigma)-np.exp(-y)-y
        elif 1+gamma*y>0:
            return -np.log(sigma)-(1+gamma*y)**(-1/gamma)+np.log(1+gamma*y)*(-1/gamma-1)
        else:
            return -1000 # ln(0)
    
    elif option == 2:
        # y[0] is max, y[1] 2nd largest
        if np.abs(gamma) < 0.0001:
            return - 2 * np.log(sigma) - np.exp(-y[1]) - y[0] - y[1] 
        elif 1+gamma*y[0]>0 and 1+gamma*y[1]>0:
            return -2 * np.log(sigma)-(1+gamma*y[1])**(-1/gamma)-np.log(1+gamma*y[0])*(1/gamma+1)-np.log(1+gamma*y[1])*(1/gamma+1)
        else:
            return -1000 # ln(0)
    
    elif option == 3 and max_only:
        # y is maximum
        if np.abs(gamma) < 0.0001:
            return - 2 * np.log(sigma) - np.exp(-y) - y
        elif 1+gamma*y>0:
            l = -2 * np.log(sigma)
            l += -(1+gamma*y)**(-1/gamma)
            l += -np.log(1+gamma*y)*(1/gamma+1)
            return l
        else:
            return -1000 # ln(0)
    
    elif option == 3 and second_only:
        spi = sigmoid(pi)
        # y is second largest
        if np.abs(gamma) < 0.0001 and 1-spi+spi*(1+gamma*y)**(-1/gamma)>0:
            return  - np.exp(-y) - y - np.log(1-spi+spi*np.exp(-y))
        elif 1+gamma*y>0 and 1-spi+spi*(1+gamma*y)**(-1/gamma)>0:
            l = -(1+gamma*y)**(-1/gamma)
            l += -np.log(1+gamma*y)*(1/gamma+1)
            l +=- np.log(1-spi+spi*(1+gamma*y)**(-1/gamma))
            return l
        else:
            return -1000 # ln(0)

    elif option == 3:
        # y[0] is max, y[1] 2nd largest
        spi = sigmoid(pi)
        if np.abs(gamma) < 0.0001 and 1-spi+spi*(1+gamma*y[1])**(-1/gamma)>0:
            return - 2 * np.log(sigma) - np.exp(-y[0]) - np.exp(-y[1]) - y[0] - y[1] - np.log(1-spi+spi*np.exp(-y[1]))
        elif 1+gamma*y[0]>0 and 1+gamma*y[1]>0 and 1-spi+spi*(1+gamma*y[1])**(-1/gamma)>0:
            l = -2 * np.log(sigma)
            l += -(1+gamma*y[0])**(-1/gamma)-(1+gamma*y[1])**(-1/gamma)
            l += -np.log(1+gamma*y[0])*(1/gamma+1)-np.log(1+gamma*y[1])*(1/gamma+1)
            l +=- np.log(1-spi+spi*(1+gamma*y[1])**(-1/gamma))
            return l
        else:
            return -1000 # ln(0)
    
# the log GEV for lists
def ll_GEV(x, gamma=0, mu=0, sigma=1, pi=1, option=1, max_only=False, second_only=False):
    g = lambda x: ll_gev(x, gamma, mu, sigma, pi, option, max_only=max_only, second_only=second_only)
    return list(map(g, x))

# PWM Estimation
def PWM_estimation(maxima):
    n = len(maxima)
    if n > 2:
        m = np.sort(maxima)
        b_0 = np.mean(maxima)
        b_1 = sum([i*m[i] for i in range(n)]) / (n*(n-1))
        b_2 = sum([i*(i-1)*m[i] for i in range(n)]) / (n*(n-1)*(n-2))
        return b_0, b_1, b_2
    else:
        print('PWM requires at least 3 maxima!')
        return np.nan, np.nan, np.nan

def PWM2GEV(b_0, b_1, b_2):
    def g1(x):
        if x == 0:
            return np.log(3)/np.log(2)
        else:
            return (3**x-1)/(2**x-1)
        
    def g2(x):
        if x == 0:
            return 1/np.log(2)
        else:
            return x/(Gamma(1-x)*(2**x-1))
        
    def g3(x):
        if x == 0:
            return - np.euler_gamma
        else:
            return (1 - Gamma(1-x)) / x
    
    def invg1(y):
        return inversefunc(g1, y_values=y).item()
    
    gamma = invg1((3*b_2-b_0)/(2*b_1-b_0))
    
    sigma = g2(gamma)*(2*b_1-b_0)

    mu = b_0 + sigma * g3(gamma)

    return gamma, mu, sigma 


def simulate_timeseries(n, distr='GEV', correlation='IID', modelparams=[0], ts=0, seed=None):
    """
        Simulates a time series based on a given correllation structure.
    """
    if seed is not None:
        np.random.seed(seed)
    # draw from GEV
    if correlation.upper() == 'IID':
        if distr == 'GEV':
            if modelparams[0] == 0:
                # gumbel case
                s = gumbel_r.rvs(size=n)
            if modelparams[0] > 0:
                # frechet case
                s = invweibull.rvs(1/modelparams[0], size=n)
            if modelparams[0] < 0:
                # weibull case
                s = weibull_max.rvs(-1/modelparams[0], size=n)
        if distr == 'GPD':
            s = genpareto.rvs(c=modelparams[0], size=n)

    elif correlation == 'ARMAX':
        # creating Frechet(1)-AMAX model
        Z = invweibull.rvs(1, size=n)
        X = [Z[0]]
        for f in Z[1:]:
            xi = np.max([ts * X[-1], (1 - ts) * f])
            X.append(xi)
        # transforming model to desired distribution
        if distr == 'GPD':
            s = genpareto.ppf(invweibull.cdf(X, c=1), c=modelparams[0])
        else:
            raise ValueError('Other distributions yet to be implemented')
        
    elif correlation == 'AR':
        Z = cauchy.rvs(size=n)
        X = [Z[0]]
        for f in Z[1:]:
            xi = np.max([ts * X[-1], (1 - ts) * f])
            X.append(xi)
        # transforming model to desired distribution
        if distr == 'GPD':
            s = genpareto.ppf(cauchy.cdf(X), c=modelparams[0])
        else:
            raise ValueError('Other distributions yet to be implemented')
        

    else:
        raise ValueError('No valid model specified')
    
    return s
    
def stride2int(stride, block_size):
    if stride == 'SBM':
        return 1
    elif stride == 'DBM':
        return int(block_size) 
    else:
        return int(stride)

def modelparams2gamma_true(distr, correllation, modelparams):
    """
    calculate, if known, the theoretical gamma resulting from model specification.
    This is especially necessary for MSE calculation.
    """
    if distr in ['GEV', 'GPD'] and correllation in ['IID', 'ARMAX', 'AR']: # NOT PROVEN FOR AR, or find source
        return modelparams[0]


def mse(gammas, gamma_true):
    if len(gammas) > 1:
        MSE = sum((np.array(gammas) - gamma_true)**2)/(len(np.array(gammas))-1)
        variance = sum((np.array(gammas) - np.mean(gammas))**2)/(len(np.array(gammas))-1)
        bias = MSE - variance
        return MSE, variance, bias
    else:
        warnings.warn('No variance can be computed on only 1 element!')
        return np.nan, np.nan, np.nan


