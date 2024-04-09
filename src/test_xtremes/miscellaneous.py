"""
test_xtremes.miscellaneous - Python sublibrary for diverse functions.

.. code-block:: python

    # Import test_xtremes
    import test_xtremes.miscellaneous as misc

    # Call an arbitray function
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

# BASIC FUNTIONS
def sigmoid(x):
    r""" Sigmoid  of x

    Notes
    -----
    Computes the sigmoid of given values.
    
    .. math::
        \sigma(x) := \frac{1}{1+\exp(-x)}
    
    Parameters
    ----------
    :param x: input, :math:`x\in\mathbb{R}`
    :type x: int, float, list or numpy.array
    :return: The sigmoid of the input
    :rtype: numpy.ndarray[float]
    """
    x = np.array(x)
    return 1/(1+np.exp(-x))

def invsigmoid(y):
    r""" Inverse Sigmoid of x

    Notes
    -----
    Computes the inverse sigmoid :math:`\sigma^{-1}` of given values, where :math:`\sigma` is defined as
   
    .. math::
        \sigma(x) := 1/(1+\exp(-x)).
    
    Parameters
    ----------
    :param x: input, :math:`x\in [0, 1]`
    :type x: int, float, list or numpy.array
    :return: The inverse sigmoid of the input
    :rtype: numpy.ndarray[float]
    :raise test_xtremes.miscellaneous.ValueError: If values outside [0,1] are given as input
    """
    if y < 0 or y > 1:
        return ValueError('Sigmoid only maps to (0, 1)!')
    elif y < np.exp(-30):
        return -30 # exp(-30)=0
    elif y > 1-np.exp(30):
        return 30 # exp(30)=infty
    else:
        return inversefunc(sigmoid, y_values=y).item()


def mse(gammas, gamma_true):
    r""" Mean Squared Error for shape param

    Notes
    -----
    Computes the mean squared error, variance and bias of a set of estimators given the true 
    (theoretical) value. This function is originally intended for estimating the GEV shape parameter
    :math:`gamma`, but works for other estimators just as fine
   
    .. math::
        \mathrm{MSE}(\hat\gamma) &:= \frac{1}{n-1}\sum_{i=1}^n \left(\hat\gamma_i-\gamma\right)^2 \\
        \mathrm{Var}(\hat\gamma) &:= \frac{1}{n-1}\sum_{i=1}^n \left(\hat\gamma_i-\overline\gamma\right)^2 \\
        \mathrm{Bias}(\hat\gamma) &:= \frac{n}{n-1} \left(\gamma-\overline\gamma\right)^2.
    
    Here, :math:`\overline\gamma` denotes the mean 

    .. math::
        \overline\gamma := \frac{1}{n}\sum_{i=1}^n \gamma_i .
    
    Also note that

    .. math::
        \mathrm{MSE} := \mathrm{Bias} + \mathrm{Var}. 

    Parameters
    ----------
    :param gammas: list of estimated values
    :param gamma_true: true / theoretical parameter
    :type gammas: list or numpy.array
    :type gamma_true: int or float
    :return: MSE, variance and bias 
    :rtype: tuple[float]
    :raise test_xtremes.miscellaneous.warning: If len(gammas)==1. nans are returned.
    """
    if len(gammas) > 1:
        MSE = sum((np.array(gammas) - gamma_true)**2)/(len(np.array(gammas))-1)
        variance = sum((np.array(gammas) - np.mean(gammas))**2)/(len(np.array(gammas))-1)
        bias = MSE - variance
        return MSE, variance, bias
    else:
        warnings.warn('No variance can be computed on only 1 element!')
        return np.nan, np.nan, np.nan

def GEV_cdf(x, gamma=0, mu=0, sigma=1, theta=1):
    r""" CDF of the GEV

    Notes
    -----
    Computes the cumulative density function of the Generalized Extreme Value distribution
   
    .. math::
        G_{\gamma,\mu,\sigma}(x):= \exp\left(-\left(1+\gamma \frac{x-\mu}{\sigma}\right)^{-1/\gamma}\right).
    
    For :math:`\gamma=0`, this term can be interpreted as the limit :math:`\lim_{\gamma\to 0}`.:

    .. math::
        G_{0,\mu,\sigma}(x):= \exp\left(-\exp\left(-\frac{x-\mu}{\sigma}\right)\right).
    
        
    This function also allows the usage of an extremal index, another parameter relevant when
    dealing with stationary time series and its extreme values.

    .. math::
        G_{\gamma,\mu,\sigma,\vartheta}(x):= \exp\left(-\vartheta\left(1+\gamma \frac{x-\mu}{\sigma}\right)^{-1/\gamma}\right).
    
    Parameters
    ----------
    :param x: input, GEV argument :math:`x\in \mathbb{R}`
    :type x: int, float, list or numpy.array
    :param gamma: input, GEV shape parameter :math:`\gamma\in \mathbb{R}`
    :type gamma: int, float, list or numpy.array
    :param mu: input, GEV location parameter :math:`\mu\in \mathbb{R}`
    :type mu: int, float, list or numpy.array
    :param sigma: input, GEV scale parameter :math:`\sigma>0`
    :type sigma: int, float, list or numpy.array
    :param theta: input, extremal index :math:`\vartheta\in [0, 1]`
    :type theta: int, float, list or numpy.array
    :return: The inverse sigmoid of the input
    :rtype: numpy.ndarray[float] or float
    """
    x = np.array(x)
    y = (x-mu)/sigma
    if gamma == 0:
        tau = -np.exp(-y)
    else:
        tau = (1+gamma*x)**(-1/gamma)
    out = np.exp(-theta*tau)
    if len(out) == 1:
        return out[0]
    else:
        return out



def GEV_pdf(x, gamma=0, mu=0, sigma=1):
    r""" PDF of the GEV

    Notes
    -----
    Computes the probability density function of the Generalized Extreme Value distribution
   
    .. math::
        g(x) := \exp\left(-\left(1+\gamma \frac{x-\mu}{\sigma}\right)^{-1/\gamma}\right)\cdot\left(1+\gamma \frac{x-\mu}{\sigma}\right)^{-1-1/\gamma}/\sigma
    
    Parameters
    ----------
    :param x: input, GEV argument :math:`x\in \mathbb{R}`
    :type x: int, float, list or numpy.array
    :param gamma: input, GEV shape parameter :math:`\gamma\in \mathbb{R}`
    :type gamma: int, float, list or numpy.array
    :param mu: input, GEV location parameter :math:`\mu\in \mathbb{R}`
    :type mu: int, float, list or numpy.array
    :param sigma: input, GEV scale parameter :math:`\sigma>0`
    :type sigma: int, float, list or numpy.array
    :return: The inverse sigmoid of the input
    :rtype: numpy.ndarray[float] or float
    """
    x = np.array(x)
    y = (x-mu)/sigma #standard
    if gamma == 0:
        out = np.exp(-np.exp(-y))*np.exp(-y)/sigma
    elif 1+gamma*y>0:
        out = np.exp(-(1+gamma*y)**(-1/gamma))*(1+gamma*y)**(-1/gamma-1)/sigma
    else:
        out = np.zeros_like(x)
    if len(out) == 1:
        return out[0]
    else:
        return out


def GEV_ll(x, gamma=0, mu=0, sigma=1):
    r""" log-likelihood of the GEV

    Notes
    -----
    Computes the log-likelihood function of the Generalized Extreme Value distribution
   
    .. math::
        l(x) := -\left(1+\gamma \frac{x-\mu}{\sigma}\right)^{-1/\gamma}-\frac{\gamma+1}{\gamma}\log\left(1+\gamma \frac{x-\mu}{\sigma}\right)-\log\sigma
    
    Parameters
    ----------
    :param x: input, GEV argument :math:`x\in \mathbb{R}`
    :type x: int, float, list or numpy.array
    :param gamma: input, GEV shape parameter :math:`\gamma\in \mathbb{R}`
    :type gamma: int, float, list or numpy.array
    :param mu: input, GEV location parameter :math:`\mu\in \mathbb{R}`
    :type mu: int, float, list or numpy.array
    :param sigma: input, GEV scale parameter :math:`\sigma>0`
    :type sigma: int, float, list or numpy.array
    :return: The inverse sigmoid of the input
    :rtype: numpy.ndarray[float] or float
    """
    x = np.array(x)
    sigma = np.abs(sigma)
    y = (x-mu)/sigma #standard
    if np.abs(gamma) < 0.01:
        out = -np.log(sigma)-np.exp(-y)-y
    elif 1+gamma*y>0:
        out = -np.log(sigma)-(1+gamma*y)**(-1/gamma)+np.log(1+gamma*y)*(-1/gamma-1)
    else:
        out = -1000 * np.ones_like(x)
    if len(out) == 1:
        return out[0]
    else:
        return out

# PWM Estimation
def PWM_estimation(maxima):
    r""" PWM Estimation of GEV params

    Notes
    -----
    Computes Probability Weighted Moment estimators on given block maxima, as introduced in :cite:`Greenwood1979`.
   
    .. math::
        \sigma(x) := 1/(1+\exp(-x)).
    
    Parameters
    ----------
    :param x: input, :math:`x\in [0, 1]`
    :type x: int, float, list or numpy.array
    :return: The inverse sigmoid of the input
    :rtype: numpy.ndarray[float]
    :raise test_xtremes.miscellaneous.ValueError: If values outside [0,1] are given as input
    """
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
    r""" Inverse Sigmoid of x

    Notes
    -----
    Computes the inverse sigmoid :math:`\sigmoid^{-1}` of given values, where :math:`sigmoid` is defined as
   
    .. math::
        \sigma(x) := 1/(1+\exp(-x)).
    
    Parameters
    ----------
    :param x: input, :math:`x\in [0, 1]`
    :type x: int, float, list or numpy.array
    :return: The inverse sigmoid of the input
    :rtype: numpy.ndarray[float]
    :raise test_xtremes.miscellaneous.ValueError: If values outside [0,1] are given as input
    """
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
    r""" Inverse Sigmoid of x

    Notes
    -----
    Computes the inverse sigmoid :math:`\sigmoid^{-1}` of given values, where :math:`sigmoid` is defined as
   
    .. math::
        \sigma(x) := 1/(1+\exp(-x)).
    
    Parameters
    ----------
    :param x: input, :math:`x\in [0, 1]`
    :type x: int, float, list or numpy.array
    :return: The inverse sigmoid of the input
    :rtype: numpy.ndarray[float]
    :raise test_xtremes.miscellaneous.ValueError: If values outside [0,1] are given as input
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
    r""" Inverse Sigmoid of x

    Notes
    -----
    Computes the inverse sigmoid :math:`\sigmoid^{-1}` of given values, where :math:`sigmoid` is defined as
   
    .. math::
        \sigma(x) := 1/(1+\exp(-x)).
    
    Parameters
    ----------
    :param x: input, :math:`x\in [0, 1]`
    :type x: int, float, list or numpy.array
    :return: The inverse sigmoid of the input
    :rtype: numpy.ndarray[float]
    :raise test_xtremes.miscellaneous.ValueError: If values outside [0,1] are given as input
    """
    if stride == 'SBM':
        return 1
    elif stride == 'DBM':
        return int(block_size) 
    else:
        return int(stride)

def modelparams2gamma_true(distr, correllation, modelparams):
    r""" Inverse Sigmoid of x

    Notes
    -----
    Computes the inverse sigmoid :math:`\sigmoid^{-1}` of given values, where :math:`sigmoid` is defined as
   
    .. math::
        \sigma(x) := 1/(1+\exp(-x)).
    
    Parameters
    ----------
    :param x: input, :math:`x\in [0, 1]`
    :type x: int, float, list or numpy.array
    :return: The inverse sigmoid of the input
    :rtype: numpy.ndarray[float]
    :raise test_xtremes.miscellaneous.ValueError: If values outside [0,1] are given as input
    """
    if distr in ['GEV', 'GPD'] and correllation in ['IID', 'ARMAX', 'AR']: # NOT PROVEN FOR AR, or find source
        return modelparams[0]



