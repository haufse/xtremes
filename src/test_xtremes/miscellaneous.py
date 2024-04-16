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
    """
    Compute the sigmoid function for the given input.

    Parameters
    ----------
    :param x: array_like
        The input value or array.

    Returns
    -------
    :return: numpy.ndarray
        The sigmoid of the input array. It has the same shape as `x`.

    Notes
    -----
    - The sigmoid function is defined as:

      .. math::
         \text{sigmoid}(x) = \frac{1}{1 + e^{-x}}

    - It maps any real-valued number to the range (0, 1).

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([-2, -1, 0, 1, 2])
    >>> sigmoid(x)
    array([0.11920292, 0.26894142, 0.5       , 0.73105858, 0.88079708])
    """
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

def invsigmoid(y):
    """
    Compute the inverse sigmoid function for the given input.

    Parameters
    ----------
    :param y: float or array_like
        The input value or array, representing probabilities in the range [0, 1].

    Returns
    -------
    :return: float or numpy.ndarray
        The inverse sigmoid of the input value or array.

    Raises
    ------
    ValueError
        If the input value is outside the range [0, 1].

    Notes
    -----
    - The inverse sigmoid function finds the input value(s) that would produce the given output probability.
    - It is the inverse of the sigmoid function: `invsigmoid(sigmoid(x)) = x`.

    Example
    -------
    >>> import numpy as np
    >>> y = np.array([0.1, 0.5, 0.9])
    >>> invsigmoid(y)
    array([-1.3652469 ,  0.        ,  1.3652469 ])
    """
    if y < 0 or y > 1:
        raise ValueError('Sigmoid only maps to (0, 1)!')
    elif y < np.exp(-30):
        return -30  # exp(-30) = 0
    elif y > 1 - np.exp(30):
        return 30  # exp(30) = infinity
    else:
        return inversefunc(sigmoid, y_values=y).item()


def mse(gammas, gamma_true):
    """
    Compute Mean Squared Error, Variance, and Bias of estimators.

    Notes
    -----
    Computes the Mean Squared Error (MSE), Variance, and Bias of a set of estimators
    given the true (theoretical) value. This function is intended for estimating the GEV
    shape parameter :math:`\gamma`, but works for other estimators as well.

    .. math::

        \mathrm{MSE}(\hat\gamma) &:= \frac{1}{n-1}\sum_{i=1}^n \left(\hat\gamma_i-\gamma\right)^2 \\
        \mathrm{Var}(\hat\gamma) &:= \frac{1}{n-1}\sum_{i=1}^n \left(\hat\gamma_i-\overline\gamma\right)^2 \\
        \mathrm{Bias}(\hat\gamma) &:= \frac{n}{n-1} \left(\gamma-\overline\gamma\right)^2

    Here, :math:`\overline\gamma` denotes the mean.

    .. math::

        \overline\gamma := \frac{1}{n}\sum_{i=1}^n \gamma_i

    Also note that:

    .. math::

        \mathrm{MSE} := \mathrm{Bias} + \mathrm{Var}

    Parameters
    ----------
    :param gammas: array_like
        Estimated values.
    :param gamma_true: int or float
        True/theoretical parameter.
    :type gammas: list or numpy.ndarray
    :type gamma_true: int or float

    Returns
    -------
    :return: tuple[float]
        MSE, variance, and bias.

    Raises
    ------
    :raise test_xtremes.miscellaneous.warning: If len(gammas)==1. NaNs are returned.
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
    """
    Compute the Cumulative Density Function (CDF) of the Generalized Extreme Value distribution.

    Notes
    -----
    Computes the cumulative density function of the Generalized Extreme Value distribution:

    .. math::

        G_{\gamma,\mu,\sigma}(x) = \exp\left(-\left(1+\gamma \frac{x-\mu}{\sigma}\right)^{-1/\gamma}\right).

    For :math:`\gamma=0`, the term can be interpreted as the limit :math:`\lim_{\gamma\to 0}`:

    .. math::

        G_{0,\mu,\sigma}(x) = \exp\left(-\exp\left(-\frac{x-\mu}{\sigma}\right)\right).

    This function also allows the usage of an extremal index, another parameter relevant when
    dealing with stationary time series and its extreme values:

    .. math::

        G_{\gamma,\mu,\sigma,\vartheta}(x) = \exp\left(-\vartheta\left(1+\gamma \frac{x-\mu}{\sigma}\right)^{-1/\gamma}\right).

    Parameters
    ----------
    :param x: int, float, list or numpy.ndarray
        GEV argument :math:`x\in \mathbb{R}`.
    :param gamma: int, float, list or numpy.ndarray, optional
        GEV shape parameter :math:`\gamma\in \mathbb{R}`. Default is 0.
    :param mu: int, float, list or numpy.ndarray, optional
        GEV location parameter :math:`\mu\in \mathbb{R}`. Default is 0.
    :param sigma: int, float, list or numpy.ndarray, optional
        GEV scale parameter :math:`\sigma>0`. Default is 1.
    :param theta: int, float, list or numpy.ndarray, optional
        Extremal index :math:`\vartheta\in [0, 1]`. Default is 1.

    Returns
    -------
    :return: numpy.ndarray or float
        The Cumulative Density Function values corresponding to the input `x`.

    """
    x = np.array(x)
    y = (x-mu)/sigma
    if gamma == 0:
        tau = -np.exp(-y)
    else:
        tau = (1+gamma*x)**(-1/gamma)
    out = np.exp(-theta*tau)
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
    return out

# PWM Estimation
def PWM_estimation(maxima):
    r""" PWM Estimation of GEV params

    Notes
    -----
    Computes the first three Probability Weighted Moments :math:`\beta_0,\beta_1,\beta_2`
    on given block maxima, as introduced in :cite:`Greenwood1979`.

    Let :math:`M_{(1)} \leq M_{(2)} \leq \cdots \leq M_{(n)}` be increasingly sorted block maxima.
    Then the first three PWMs are defined as
   
    .. math::
        \beta_0 &:= \frac{1}{n}\sum_{i=1}^n M_{(i)},\\
        \beta_1 &:= \frac{1}{n(n-1)}\sum_{i=1}^n (i-1) M_{(i)},\\
        \beta_2 &:= \frac{1}{n(n-1)(n-2)}\sum_{i=1}^n (i-1)(i-2)M_{(i)},\\
    
    Parameters
    ----------
    :param x: sequence of maxima
    :type x: list or numpy.array, `len(x)` :math:`\geq3`
    :return: the first three PWMs
    :rtype: tuple[float]
    :raise test_xtremes.miscellaneous.warning: If less than three maxima are given as input
    """
    n = len(maxima)
    if n > 2:
        m = np.sort(maxima)
        b_0 = np.mean(maxima)
        b_1 = sum([i*m[i] for i in range(n)]) / (n*(n-1))
        b_2 = sum([i*(i-1)*m[i] for i in range(n)]) / (n*(n-1)*(n-2))
        return b_0, b_1, b_2
    else:
        warnings.warn('PWM requires at least 3 maxima!')
        return np.nan, np.nan, np.nan

def PWM2GEV(b_0, b_1, b_2):
    r"""GEV params from PWM 

    Notes
    -----
    Computes estimators for the parameters of the GEV distribution from given PWM estimators.
    As shown in :cite:`Hosking1985`, they follow the relationship
   
    .. math::
        \gamma &= g_1^{-1}\left(\frac{3\beta_2-\beta_0}{2\beta_1-\beta_0}\right) \\
        \sigma &= g_2(\gamma)\cdot (2\beta_1-\beta_0) \\
        \mu &= \beta_0+\sigma\cdot g_3(\gamma),
    
    where
    
    .. math::
        g_1(\gamma) &:= \frac{3^\gamma-1}{2^\gamma-1} \\
        g_2(\gamma) &:= \frac{\gamma}{\Gamma(1-\gamma)(2^\gamma-1)} \\
        g_3(\gamma) &:= \frac{1-\Gamma(1-\gamma)}{\gamma}.

    Note that :math:`\Gamma` denotes the gamma function. The values for :math:`g_\bullet(0)` 
    are defined by continuity to result in  

    .. math::
        g_1(0)= \frac{\log 3}{\log 2}, \qquad g_2(0) = \frac{1}{\log 2}, \qquad g_3(0) = -\gamma_\mathrm{EM}
    
    with :math:`\gamma_\mathrm{EM}` being the Euler-Mascheroni constant.

    Parameters
    ----------
    :param b_0, b_1, b_2: PWM estimators, as output from ``PWM_estimators``
    :type b_0, b_1, b_2: int or float
    :return: Estimators for the GEV params
    :rtype: tuple[float]
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
    r""" Simulate a Time Series for GEV

    Notes
    -----
    This function allows to simulate three different kind of time series. 

    The most basic time series is the IID case, in which there is no temporal dependence.
    The distribution from which the random variables are drawn can be chosen via ``distr``, 
    respective model parameters are passed via ``modelparams``.

    One model for a stationary time series with temporal dependence is the ARMAX model.
    here, the next value is computed via
    
    .. math::
        X_0&:=Z_0 \\
        X_i&:=\max\{\alpha\cdot X_{i-1}, (1-\alpha)\cdot Z_i\}
    
    where :math:`Z_i` is drawn from a GPD distribution. The time series parameter :math:`\alpha`
    is passed via ``ts``. By specifying the ``distr``, the values are afterwards transformed to 
    follow the specified distribution.

    The second model for a stationary time series with temporal dependence is the AR model.
    here, the next value is computed via
    
    .. math::
        X_0&:=Z_0 \\
        X_i&:=\max\{\alpha\cdot X_{i-1} + (1-\alpha)\cdot Z_i\}
    
    where :math:`Z_i` is drawn from a Cauchy distribution. The time series parameter :math:`\alpha`
    is passed via ``ts``. By specifying the ``distr``, the values are afterwards transformed to 
    follow the specified distribution.

    See also
    --------
    It is highly recommended to see the associated Tutorial.
    
    Parameters
    ----------
    :param n: length of time series to simulate
    :type n: int
    :param distr: distribution to draw from
    :type distr: str
    :param correllation: correllation to specify, choose from ``['IID', 'ARMAX', 'AR']``
    :type correllation: str
    :param modelparams: parameters belonging to ``distr``
    :type modelparams: list
    :param ts: time series parameter :math:`\alpha\in[0,1]`
    :type ts: float
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
    r""" Integer from Stride

    Notes
    -----
    This function is a utility when handling Block maxima (disjoint, sliding, striding).
    Apart from giving the stride directly, it is handy to have the additional options
    `'SBM'` and `'DBM'` for sliding and disjoint BM, respectively. This funtion converts exactly this.

    Parameters
    ----------
    :param stride: stride to be converted
    :type stride: int or str
    :param block_size: block size for conversion, ignored if ``stride=='SBM'``
    :type block_size: int
    :return: The converted stride
    :rtype: int
    """
    if stride == 'SBM':
        return 1
    elif stride == 'DBM':
        return int(block_size) 
    else:
        return int(stride)

def modelparams2gamma_true(distr, correllation, modelparams):
    r""" Extract gamma_true from model params

    Notes
    -----
    For some few models, it is possible to extract the true value
    of gamma theoretically. Whenever this is possible, the conversion
    should be subject to this function.

    Parameters
    ----------
    :param distr: valid distribution type
    :type distr: str
    :param correllation: valid correllation type, currently ``['IID', 'ARMAX', 'AR']``
    :type correllation: str
    :param modelparams: valid modelparam
    :type modelparams: list
    :return: gamma_true, if applicable
    :rtype: numpy.ndarray[float]
    """
    if distr in ['GEV', 'GPD'] and correllation in ['IID', 'ARMAX', 'AR']: # NOT PROVEN FOR AR, or find source
        return modelparams[0]



