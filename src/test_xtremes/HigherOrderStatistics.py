import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import genextreme, invweibull, weibull_max, gumbel_r
from scipy.special import gamma as Gamma
from pynverse import inversefunc
import asyncio
import pickle
import warnings
import test_xtremes.miscellaneous as misc

# FUNCTIONS

def extract_BM(timeseries, block_size=10, stride='DBM'):
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
    n = len(timeseries)
    r = block_size
    p = misc.stride2int(stride, block_size)
    out = [np.max(timeseries[i*p:i*p+r]) for i in range((n-r)//p+1)]
    return np.array(out)

def extract_HOS(timeseries, orderstats=2, block_size=10, stride='DBM'):
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
    n = len(timeseries)
    r = block_size
    p = misc.stride2int(stride, block_size)
    out = [np.sort(timeseries[i*p:i*p+r])[-orderstats:] for i in range((n-r)//p+1)]
    return np.array(out)
    

def cost_function(high_order_stats, option=1, estimate_pi=False, pi0=1):
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
    # define cost function based on option
    maxima = high_order_stats.T[-1]
    secondlargest = high_order_stats.T[-2]
    if option == 1:       
        def cost(params):
            gamma, mu, sigma, pi = params     
            uni, count = np.unique(maxima, return_counts=True)   
            cst = - np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option), count)
            return cst
        
    if option == 2:
        def cost(params):
            gamma, mu, sigma, pi = params     
            uni, count = np.unique(high_order_stats, return_counts=True, axis=0)   
            cst = - np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option), count)
            return cst
    
    if option == 3:
        def cost(params):
            gamma, mu, sigma, pi = params
            if not estimate_pi:
                pi = pi0
            # add likelihood for maxima
            uni, count = np.unique(maxima, return_counts=True)   
            cst = - np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option, max_only=True), count)
            # add likelihood for second largest
            uni, count = np.unique(secondlargest, return_counts=True)   
            cst -=  np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option, second_only=True), count)
            return cst
        
    return cost

def automatic_parameter_initialization(PWM_estimators, corr, ts=0.5):
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
    initParams = np.array(PWM_estimators)
    # pi parameter from theory
    if corr == 'ARMAX':
        pis = np.ones(shape=(1,len(PWM_estimators))) * misc.invsigmoid(1-ts)
    elif corr == 'IID':
        pis = np.ones(shape=(1,len(PWM_estimators)))
    else:
        pis = np.ones(shape=(1,len(PWM_estimators)))
    return np.append(initParams.T, pis, axis=0).T


# parallel running ML estimations
def background(f):
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
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def run_ML_estimation(file, corr='IID', gamma_true=0, block_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], stride='SBM', option=1, estimate_pi=False):
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
    with open(file, 'rb') as f:
        timeseries = pickle.load(f)
    out = {}
    for block_size in block_sizes:
        print(f'Running block size {block_size}')
        series = timeseries[corr][gamma_true]
        series.get_blockmaxima(block_size=block_size, stride=stride)
        highorderstats = HighOrderStats(series)
        highorderstats.get_ML_estimation(option=option, estimate_pi=estimate_pi)
        out[block_size] = highorderstats
    res = {gamma_true: out}
    return res

def run_multiple_ML_estimations(file, corr='IID', gamma_trues=np.arange(-4, 5, 1)/10, block_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], stride='SBM', option=1, estimate_pi=False):
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
    loop = asyncio.get_event_loop()
    looper = asyncio.gather(*[run_ML_estimation(file, corr, gamma_true, block_sizes, stride, option, estimate_pi) for gamma_true in gamma_trues])
    results = loop.run_until_complete(looper) 
    # merge list of dicts to single dict
    if len(results) > 1:
        out_dict = results[0]
        for i in range(1, len(results)):
            out_dict = out_dict | results[i]
    return out_dict

# CLASSES

class PWM_estimators:
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
    def __init__(self, TimeSeries):
        # TimeSeries object needs blockmaxima
        if TimeSeries.blockmaxima == [] and TimeSeries.high_order_stats == []:
            raise ValueError('Caluclate block maxima or high order stats first!')
        if TimeSeries.blockmaxima != []:
            self.blockmaxima = TimeSeries.blockmaxima
        else:
            self.blockmaxima = np.array(TimeSeries.high_order_stats).T[-1]
        self.values = []
        self.statistics = {}
    
    def __len__(self):
        return len(self.values)
    
    def get_PWM_estimation(self):
        for bms in self.blockmaxima:
            b0, b1, b2 = misc.PWM_estimation(bms) 
            gamma, mu, sigma = misc.PWM2GEV(b0, b1, b2)
            self.values.append([gamma, mu, sigma])
        self.values = np.array(self.values)
    
    def get_statistics(self, gamma_true):
        gammas = self.values.T[0]
        mus = self.values.T[1]
        sigmas = self.values.T[2]
        if len(gammas) > 1:
            # gamma-related statistics
            MSE = sum((np.array(gammas) - gamma_true)**2)/(len(np.array(gammas))-1)
            mean = np.mean(gammas)
            variance = sum((np.array(gammas) - mean)**2)/(len(np.array(gammas))-1)
            bias = MSE - variance
            self.statistics['mean'] = mean
            self.statistics['variance'] = variance
            self.statistics['bias'] = bias
            self.statistics['mse'] = MSE
            # mu and sigma-related statistics
            mu_mean = np.mean(mus)
            mu_variance = sum((np.array(mus) - mu_mean)**2)/(len(np.array(mus))-1)
            sigma_mean = np.mean(sigmas)
            sigma_variance = sum((np.array(sigmas) - sigma_mean)**2)/(len(np.array(sigmas))-1)
            self.statistics['mu_mean'] = mu_mean
            self.statistics['mu_variance'] = mu_variance
            self.statistics['sigma_mean'] = sigma_mean
            self.statistics['sigma_variance'] = sigma_variance

        else:
            warnings.warn('No variance can be computed on only one element!')

class ML_estimators:
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
    def __init__(self, TimeSeries):
        # TimeSeries object needs high order statistics
        if TimeSeries.high_order_stats == []:
            raise ValueError('Caluclate high order statistics first!')
        self.TimeSeries = TimeSeries
        self.high_order_stats = TimeSeries.high_order_stats
        self.values = []
        self.statistics = {}
    
    def __len__(self):
        return len(self.values)

    def get_ML_estimation(self, PWM_estimators=None, initParams = 'auto', option=1, estimate_pi=False):
        if initParams == 'auto' and PWM_estimators==None:
            raise ValueError('Automatic calculation of initParams needs PWM_estimators!')
        elif initParams == 'auto':
            initParams = automatic_parameter_initialization(PWM_estimators.values, 
                                                            self.TimeSeries.corr, 
                                                            ts=self.TimeSeries.ts)
        for i, ho_stat in enumerate(self.high_order_stats):
            cost = cost_function(ho_stat, option=option, estimate_pi=estimate_pi, pi0=initParams[i][-1])
            results = misc.minimize(cost, initParams[i], method='Nelder-Mead')
    
            gamma, mu, sigma, pi = results.x
            
            self.values.append([gamma, mu, sigma, pi])
        self.values = np.array(self.values)
    
    def get_statistics(self, gamma_true):
        gammas = self.values.T[0]
        mus = self.values.T[1]
        sigmas = self.values.T[2]
        pis = self.values.T[3]
        if len(gammas) > 1:
            # gamma-related statistics
            MSE = sum((np.array(gammas) - gamma_true)**2)/(len(np.array(gammas))-1)
            mean = np.mean(gammas)
            variance = sum((np.array(gammas) - mean)**2)/(len(np.array(gammas))-1)
            bias = MSE - variance
            self.statistics['mean'] = mean
            self.statistics['variance'] = variance
            self.statistics['bias'] = bias
            self.statistics['mse'] = MSE
            # mu and sigma-related statistics
            mu_mean = np.mean(mus)
            mu_variance = sum((np.array(mus) - mu_mean)**2)/(len(np.array(mus))-1)
            sigma_mean = np.mean(sigmas)
            sigma_variance = sum((np.array(sigmas) - sigma_mean)**2)/(len(np.array(sigmas))-1)
            pi_mean = np.mean(pis)
            pi_variance = sum((np.array(pis) - pi_mean)**2)/(len(np.array(pis))-1)
            self.statistics['mu_mean'] = mu_mean
            self.statistics['mu_variance'] = mu_variance
            self.statistics['sigma_mean'] = sigma_mean
            self.statistics['sigma_variance'] = sigma_variance
            self.statistics['pi_mean'] = pi_mean
            self.statistics['pi_variance'] = pi_variance

        else:
            warnings.warn('No variance can be computed on only one element!')
            

class TimeSeries:
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
    def __init__(self, n, distr='GEV', correlation='IID', modelparams=[0], ts=0):
        self.timeseries = []
        self.distr = distr
        self.corr = correlation
        self.modelparams = modelparams
        self.ts = ts
        self.len = n
        self.blockmaxima = []
        self.high_order_stats = []
    
    def __len__(self):
        return self.len
    
    def simulate(self, rep=10, seeds='default'):
        self.reps = 10
        if seeds == 'default':
            for i in range(rep):
                series = misc.simulate_timeseries(self.len,
                                             distr=self.distr, 
                                             correlation=self.corr, 
                                             modelparams=self.modelparams, 
                                             ts=self.ts, 
                                             seed=i)
                self.timeseries.append(series)
        elif seeds == None:
            for i in range(rep):
                series = misc.simulate_timeseries(self.len,
                                             distr=self.distr, 
                                             correlation=self.corr, 
                                             modelparams=self.modelparams, 
                                             ts=self.ts, 
                                             seed=None)
                self.timeseries.append(series)
        else:
            if not hasattr(seeds, '__len__'):
                # handles the case of seeds being an integer and rep=1
                seeds = [seeds]
            if len(seeds) != rep:
                raise ValueError('Number of given seeds does not match repititions!')
            else:
                for i in range(rep):
                    series = misc.simulate_timeseries(self.len,
                                                distr=self.distr, 
                                                correlation=self.corr, 
                                                modelparams=self.modelparams, 
                                                ts=self.ts, 
                                                seed=seeds[i])
                    self.timeseries.append(series)
        
    def get_blockmaxima(self, block_size=2, stride='DBM', rep=10):
        if self.timeseries == []:
            warnings.warn('TimeSeries was not simulated yet, I will do it for you. For more flexibility, simulate first!')
            self.simulate(rep=rep)
        self.block_size = block_size
        self.stride = stride
        for series in self.timeseries:
            bms = extract_BM(series, block_size=block_size, stride=stride)
            self.blockmaxima.append(bms)
    

    def get_HOS(self, orderstats = 2, block_size=2, stride='DBM', rep=10):
        if self.timeseries == []:
            warnings.warn('TimeSeries was not simulated yet, I will do it for you. For more flexibility, simulate first!')
            self.simulate(rep=rep)
        self.block_size = block_size
        self.stride = stride
        for series in self.timeseries:
            hos = extract_HOS(series, orderstats=orderstats, block_size=block_size, stride=stride)
            self.high_order_stats.append(hos)

class HighOrderStats:
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
    def __init__(self, TimeSeries):
        TimeSeries.get_HOS()
        self.TimeSeries = TimeSeries
        self.high_order_stats = TimeSeries.high_order_stats
        self.blockmaxima = [ho_stat.T[-1] for ho_stat in self.high_order_stats]
        self.gamma_true = misc.modelparams2gamma_true(TimeSeries.distr, TimeSeries.corr, TimeSeries.modelparams)
        self.PWM_estimators = PWM_estimators(TimeSeries=self.TimeSeries)
        self.ML_estimators = ML_estimators(TimeSeries=self.TimeSeries)
    
    def get_PWM_estimation(self):
        self.PWM_estimators.get_PWM_estimation()
        self.PWM_estimators.get_statistics(gamma_true=self.gamma_true)
        
    def get_ML_estimation(self, initParams = 'auto', option=1, estimate_pi=False):
        if self.PWM_estimators.values == [] and initParams == 'auto':
            self.get_PWM_estimation()
        self.ML_estimators.get_ML_estimation(PWM_estimators=self.PWM_estimators, initParams = initParams, option=option, estimate_pi=estimate_pi)
        self.ML_estimators.get_statistics(gamma_true=self.gamma_true)
        

