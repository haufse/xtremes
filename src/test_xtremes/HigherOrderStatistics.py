import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import genextreme, invweibull, weibull_max, gumbel_r
from scipy.special import gamma as Gamma
from pynverse import inversefunc
# import asyncio
import pickle
import warnings
import test_xtremes.miscellaneous as misc

# FUNCTIONS

# log-likelihood

def log_likelihoods(high_order_statistics, gamma=0, mu=0, sigma=1, pi=1, option=1, ts=1, corr='IID'):
    r"""Calculate the log likelihood based on the two highest order statistics in three different ways.

    Parameters
    ----------
    :param high_order_statistics: numpy array
        Array containing the two highest order statistics.
    :param gamma: float, optional
        The shape parameter for the Generalized Extreme Value (GEV) distribution. Default is 0.
    :param mu: float, optional
        The location parameter for the GEV distribution. Default is 0.
    :param sigma: float, optional
        The scale parameter for the GEV distribution. Default is 1.
    :param pi: float, optional
        The probability of the second highest order statistic being independent in option 2. Default is 1.
    :param option: int, optional
        Option for calculating the log likelihood:
        - 1: Only consider the maximum value.
        - 2: Assume the max and the second largest to be independent.
        - 3: Consider the correct joint likelihood. Default is 1.
    :param ts: float, optional
        A parameter in the ARMAX correlation mode. Default is 1.
    :param corr: str, optional
        The correlation mode. Can be 'IID' (Independent and Identically Distributed)
        or 'ARMAX' (Auto-Regressive Moving Average with eXogenous inputs). Default is 'IID'.

    Returns
    -------
    :return: float
        The calculated log likelihood.

    Notes
    -----
    - This function calculates the log likelihood based on the two highest order statistics in different ways.
    - The high_order_statistics array should contain the two highest order statistics for each observation.

    Example
    -------
    >>> hos = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.5], [0.4, 0.6]])
    >>> log_likelihoods(hos, gamma=0.5, sigma=2, option=2, corr='ARMAX')
    7.494890426732856

    """
    
    
    unique_hos, counts = np.unique(high_order_statistics, axis=0, return_counts=True)
    # split into largest and second largest
    maxima = unique_hos.T[0]
    second = unique_hos.T[1]
    sigma = np.abs(sigma)
    if sigma < 0.01:
        # ensure no division by 0
        sigma += 0.01
    # standardize both

    y_max = (maxima-mu)/sigma 
    y_sec = (second-mu)/sigma

    out = np.zeros_like(y_max)
    
    if option == 1:
        if np.abs(gamma) < 0.0001:
            out=  -np.log(sigma)-np.exp(-y_max)-y_max
        else:
            mask = (1+gamma*y_max>0)
            out[mask] = -np.log(sigma)-(1+gamma*y_max[mask])**(-1/gamma)+np.log(1+gamma*y_max[mask])*(-1/gamma-1)
            out[np.invert(mask)] = -1000

    elif option == 2:
        # forcing pi to live in [0,1]
        spi = misc.sigmoid(pi)
        if np.abs(gamma) < 0.0001:
            mask = 1-spi+spi*(1+gamma*y_sec)**(-1/gamma)>0
            out[mask] = - 2 * np.log(sigma) - np.exp(-y_max[mask]) - np.exp(-y_sec[mask]) - y_max[mask] - y_sec[mask] - np.log(1-spi+spi*np.exp(-y_sec[mask]))
            out[np.invert(mask)] = -1000 # ln(0)
        else:
            mask = np.bitwise_and(np.bitwise_and((1+gamma*y_max>0), (1+gamma*y_sec>0)), (1-spi+spi*(1+gamma*y_sec)**(-1/gamma)>0))
            out[mask] += -2 * np.log(sigma)
            out[mask] += -(1+gamma*y_max[mask])**(-1/gamma)-(1+gamma*y_sec[mask])**(-1/gamma)
            out[mask] += -np.log(1+gamma*y_max[mask])*(1/gamma+1)-np.log(1+gamma*y_sec[mask])*(1/gamma+1)
            out[mask] +=- np.log(1-spi+spi*(1+gamma*y_sec[mask])**(-1/gamma))
            out[np.invert(mask)] = -1000 # ln(0)
    
    elif option == 3:
        if np.abs(gamma) < 0.0001:
            out = - 2 * np.log(sigma) - np.exp(-second) - y_max - y_sec 
        elif corr == 'IID':
            mask = np.bitwise_and((1+gamma*y_max>0), (1+gamma*y_sec>0))
            out[mask] = -2 * np.log(sigma)-(1+gamma*y_sec[mask])**(-1/gamma)-np.log(1+gamma*y_max[mask])*(1/gamma+1)-np.log(1+gamma*y_sec[mask])*(1/gamma+1)
            out[np.invert(mask)] = -1000 # ln(0)
        elif corr == 'ARMAX':
            mask = np.bitwise_and((1+gamma*y_max>0), (1+gamma*y_sec > ts**gamma*(1+gamma*y_max)))
            out[mask] = -np.log(sigma)-(1+gamma*y_max[mask])**(-1/gamma)+np.log(1+gamma*y_max[mask])*(-1/gamma-1)
            out[np.invert(mask)] = -1000


    
    joint_ll = np.dot(out, counts)
    
    return joint_ll



def extract_BM(timeseries, block_size=10, stride='DBM'):
    r"""Extract block maxima from a given time series.

    Parameters
    ----------
    :param timeseries: list or numpy array
        The input time series data.
    :param block_size: int, optional
        The size of each block for extracting maxima. Default is 10.
    :param stride: str or int, optional
        The stride used to move the window. Can be 'DBM' (Default Block Maxima)
        or an integer specifying the stride size. Default is 'DBM'.

    Returns
    -------
    :return: numpy array
        An array of block maxima.

    Notes
    -----
    - This function divides the time series into non-overlapping blocks of size 'block_size'.
    - Within each block, the maximum value is extracted as the block maximum.
    - If the length of the time series is not divisible by 'block_size', the last block may have fewer elements.

    Example
    -------
    >>> ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    >>> extract_BM(ts, block_size=5, stride='DBM')
    array([ 5, 10, 15])
    """

    n = len(timeseries)
    r = block_size
    p = misc.stride2int(stride, block_size)
    out = [np.max(timeseries[i*p:i*p+r]) for i in range((n-r)//p+1)]
    return np.array(out)

def extract_HOS(timeseries, orderstats=2, block_size=10, stride='DBM'):
    r"""Extract high order statistics from a given time series.

    Parameters
    ----------
    :param timeseries: list or numpy array
        The input time series data.
    :param orderstats: int, optional
        The number of highest order statistics to extract. Default is 2.
    :param block_size: int, optional
        The size of each block for extracting statistics. Default is 10.
    :param stride: str or int, optional
        The stride used to move the window. Can be 'SBM' (Sliding Block Maxima),
        'DBM' (Disjoint Block Maxima), or an integer specifying the stride size. Default is 'DBM'.

    Returns
    -------
    :return: numpy array
        An array containing the extracted high order statistics.

    Notes
    -----
    - This function divides the time series into non-overlapping blocks of size 'block_size'.
    - Within each block, the highest 'orderstats' values are extracted as the high order statistics.
    - If the length of the time series is not divisible by 'block_size', the last block may have fewer elements.

    Example
    -------
    >>> ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    >>> extract_HOS(ts, orderstats=3, block_size=5, stride='DBM')
    array([[ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11],
           [12, 13, 14],
           [15, 15, 15]])
    """
    n = len(timeseries)
    r = block_size
    p = misc.stride2int(stride, block_size)
    out = [np.sort(timeseries[i*p:i*p+r])[-orderstats:] for i in range((n-r)//p+1)]
    return np.array(out)
    

def automatic_parameter_initialization(PWM_estimators, corr, ts=0.5):
    r"""
    Automatic parameter initialization for ML estimation.

    Notes
    -----
    This function is designed for initializing parameters for maximum likelihood estimation (ML) in statistical models.
    It automatically computes the probability weighted moment (PWM) estimators and adjusts them based on the specified
    correlation type ('ARMAX', 'IID', etc.). The 'ts' parameter is used to control the strength of temporal dependence
    in the model.

    Parameters
    ----------
    :param PWM_estimators: list or numpy.array
        Probability weighted moment estimators.
    :param corr: str
        Correlation type for the model. Supported values are 'ARMAX', 'IID', etc.
    :param ts: float, optional
        Time series parameter controlling the strength of temporal dependence (default is 0.5).
    :return: numpy.ndarray
        Initial parameters for ML estimation.

    See also
    --------
    misc.PWM_estimation : Function for computing probability weighted moment estimators.

    Examples
    --------
    >>> from test_xtremes import misc
    >>> PWM_est = misc.PWM_estimators(data)
    >>> init_params = automatic_parameter_initialization(PWM_est, 'ARMAX', ts=0.8)

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
    r"""Decorator for running a function in the background using asyncio.

    Parameters
    ----------
    :param f: function
        The function to be run in the background.

    Returns
    -------
    :return: function
        A wrapped version of the input function that runs in the background.

    Usage
    -----
    - Apply this decorator to a function to run it in the background.
    - The function must be a coroutine function (async def).
    - Use await keyword to call the decorated function.

    Example
    -------
    >>> @background
    >>> async def my_background_task():
    >>>     # Your background task logic goes here
    >>>     await asyncio.sleep(5)
    >>>     print("Background task completed")
    >>> # Call the decorated function using await
    >>> await my_background_task()


    Wrapper function that runs the input function in the background.

    Parameters
    ----------
    - \*args: positional arguments
        Positional arguments passed to the input function.
    - \**kwargs: keyword arguments
        Keyword arguments passed to the input function.

    Returns
    -------
    :return: result
        The result of running the input function in the background.
    """
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

# @background
def run_ML_estimation(file, corr='IID', gamma_true=0, block_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], stride='SBM', option=1, estimate_pi=False):
    r"""
    Run maximum likelihood estimation (ML) for a given time series file.

    Notes
    -----
    This function reads a time series from a file and performs ML estimation for the specified correlation type, GEV shape parameter, block sizes,
    and stride type. It iterates over each block size, extracts block maxima, computes high order statistics, and performs ML estimation.
    The results are stored in a dictionary with the GEV shape parameter as the key and ML estimation results for each block size as the value.

    Parameters
    ----------
    :param file: str
        Path to the file containing the time series data.
    :param corr: str, optional
        Correlation type for the model (default is 'IID').
    :param gamma_true: float, optional
        True value of the GEV shape parameter (default is 0).
    :param block_sizes: list, optional
        List of block sizes for extracting block maxima (default is [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]).
    :param stride: str, optional
        Stride type for block maxima extraction. Options are 'SBM' (sliding block maxima) or 'DBM' (default block maxima) (default is 'SBM').
    :param option: int, optional
        Option for ML estimation (default is 1).
    :param estimate_pi: bool, optional
        Flag indicating whether to estimate the pi parameter (default is False).
        
    Example
    -------
    >>> result = run_ML_estimation("timeseries_data.pkl", corr='ARMAX', gamma_true=0.5, block_sizes=[10, 20, 30], stride='DBM', option=2, estimate_pi=True)
    >>> print(result)
    {0.5: {10: HighOrderStats_object, 20: HighOrderStats_object, 30: HighOrderStats_object}}

    Returns
    -------
    :return: dict
        Dictionary containing ML estimation results for each block size.

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

def run_multiple_ML_estimations(file, corr='IID', gamma_trues=np.arange(-4, 5, 1)/10, block_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], stride='SBM', option=1, estimate_pi=False, parallelize=False):
    r"""
    Run multiple maximum likelihood estimations (ML) for a range of GEV shape parameter values.

    Notes
    -----
    This function performs ML estimation for a range of GEV shape parameter values specified in 'gamma_trues' and aggregates the results into a single dictionary.
    It iterates over each gamma value, calls the 'run_ML_estimation' function, and collects the results. If 'parallelize' is set to True, it runs the estimations
    concurrently using asyncio.

    Parameters
    ----------
    :param file: str
        Path to the file containing the time series data.
    :param corr: str, optional
        Correlation type for the model (default is 'IID').
    :param gamma_trues: numpy.ndarray, optional
        Array of GEV shape parameter values to perform ML estimation for (default is np.arange(-4, 5, 1)/10).
    :param block_sizes: list, optional
        List of block sizes for extracting block maxima (default is [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]).
    :param stride: str, optional
        Stride type for block maxima extraction. Options are 'SBM' (sliding block maxima) or 'DBM' (default block maxima) (default is 'SBM').
    :param option: int, optional
        Option for ML estimation (default is 1).
    :param estimate_pi: bool, optional
        Flag indicating whether to estimate the pi parameter (default is False).
    :param parallelize: bool, optional
        Flag indicating whether to parallelize the ML estimations using asyncio (default is False).

    Returns
    -------
    :return: dict
        Dictionary containing ML estimation results for each gamma value and block size.
    """
#    if parallelize:
#        loop = asyncio.get_event_loop()
#        looper = asyncio.gather(*[run_ML_estimation(file, corr, gamma_true, block_sizes, stride, option, estimate_pi) for gamma_true in gamma_trues])
#        results = loop.run_until_complete(looper) 
#    else:
    results = []
    for gamma_true in gamma_trues:
        rslt = run_ML_estimation(file, corr, gamma_true, block_sizes, stride, option, estimate_pi)
        results.append(rslt)
    # merge list of dicts to single dict
    if len(results) > 1:
        out_dict = results[0]
        for i in range(1, len(results)):
            out_dict = out_dict | results[i]
        return out_dict
    else:
        return results[0]
    

# CLASSES

class PWM_estimators:
    r"""
    Calculates Probability Weighted Moment (PWM) estimators from block maxima.

    Notes
    -----
    Probability Weighted Moments (PWMs) are used to estimate the parameters of extreme value distributions. 
    The first three PWMs (denoted as :math:`\beta_0, \beta_1, \beta_2`) are computed for each block maxima series, 
    then converted into Generalized Extreme Value (GEV) parameters (:math:`\gamma, \mu, \sigma`) using the PWM2GEV function.

    Parameters
    ----------
    :param TimeSeries: TimeSeries object
        TimeSeries object containing block maxima or high order statistics.

    Attributes
    ----------
    :attribute blockmaxima: numpy.ndarray
        Array of block maxima extracted from the TimeSeries object.
    :attribute values: numpy.ndarray
        Array containing the PWM estimators (:math:`\gamma, \mu, \sigma`) for each block maxima series.
    :attribute statistics: dict
        Dictionary containing statistics (mean, variance, bias, mse) of the PWM estimators.

    Methods
    -------
    :method get_PWM_estimation(): Compute PWM estimators for each block maxima series and convert them into GEV parameters.
    :method get_statistics(gamma_true): Compute statistics of the PWM estimators using a true gamma value.

    Raises
    ------
    :raises ValueError: If block maxima or high order statistics are not available in the TimeSeries object.

    Examples
    --------
    >>> from TimeSeries import TimeSeries
    >>> from PWM_estimators import PWM_estimators
    >>> ts = TimeSeries(data)  # initialize TimeSeries object with data
    >>> ts.get_blockmaxima(block_size=10, stride='SBM')  # extract block maxima
    >>> pwm = PWM_estimators(ts)  # initialize PWM_estimators object
    >>> pwm.get_PWM_estimation()  # compute PWM estimators
    >>> pwm.get_statistics(0.1)  # compute statistics with true gamma value 0.1
    >>> print(pwm.statistics)  # print computed statistics
    {'mean': 0.099981, 'variance': 0.000186, 'bias': -0.000074, 'mse': 0.000259,
     'mu_mean': ..., 'mu_variance': ..., 'sigma_mean': ..., 'sigma_variance': ...}

    """
    def __init__(self, TimeSeries):
        # TimeSeries object needs blockmaxima
        if TimeSeries.blockmaxima == [] and TimeSeries.high_order_stats == []:
            raise ValueError('Caluclate block maxima or high order stats first!')
        if TimeSeries.blockmaxima != []:
            self.blockmaxima = TimeSeries.blockmaxima
        else:
            self.blockmaxima = np.array(TimeSeries.high_order_stats)[:,:,-1]
        self.values = []
        self.statistics = {}
    
    def __len__(self):
        return len(self.values)
    
    def get_PWM_estimation(self):
        r"""
        Compute PWM estimators for each block maxima series and convert them into GEV parameters.
        """
        for bms in self.blockmaxima:
            b0, b1, b2 = misc.PWM_estimation(bms) 
            gamma, mu, sigma = misc.PWM2GEV(b0, b1, b2)
            self.values.append([gamma, mu, sigma])
        self.values = np.array(self.values)
    
    def get_statistics(self, gamma_true):
        r"""
        Compute statistics of the PWM estimators using a true gamma value.

        Parameters
        ----------
        :param gamma_true: float
            True gamma value for calculating statistics.
        """
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
    r"""
    Maximum Likelihood Estimators (MLE) for GEV parameters.

    This class calculates Maximum Likelihood Estimators (MLE) for the parameters of the Generalized Extreme Value (GEV) distribution using the method of maximum likelihood estimation.
    
    Parameters
    ----------
    TimeSeries : TimeSeries
        The TimeSeries object containing the data for which MLE estimators will be calculated.

    Attributes
    ----------
    values : numpy.ndarray
        An array containing the MLE estimators for each set of high order statistics.
    statistics : dict
        A dictionary containing statistics computed from the MLE estimators.

    Methods
    -------
    __len__()
        Returns the number of MLE estimators calculated.
    get_ML_estimation(PWM_estimators=None, initParams='auto', option=1, estimate_pi=False)
        Computes the MLE estimators for the GEV parameters.
    get_statistics(gamma_true)
        Computes statistics from the MLE estimators.

    Examples
    --------
    >>> import numpy as np
    >>> from your_module import TimeSeries, ML_estimators, PWM_estimators

    # Example data
    >>> blockmaxima_data = np.random.normal(loc=10, scale=2, size=100)
    >>> high_order_stats_data = np.random.normal(loc=5, scale=1, size=(100, 3))

    # Create TimeSeries object
    >>> ts = TimeSeries(blockmaxima=blockmaxima_data, high_order_stats=high_order_stats_data, corr='IID', ts=0.5)

    # Calculate PWM estimators
    >>> pwm = PWM_estimators(ts)
    >>> pwm.get_PWM_estimation()

    # Initialize ML_estimators object
    >>> ml = ML_estimators(ts)

    # Compute ML estimators
    >>> ml.get_ML_estimation(PWM_estimators=pwm)

    # Compute statistics
    >>> ml.get_statistics(gamma_true=0.1)

    # Print ML estimators
    >>> print("ML Estimators:")
    >>> print(ml.values)

    # Print statistics
    >>> print("\nStatistics:")
    >>> print(ml.statistics)
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
        r"""
        Compute ML estimators for each high order statistics series.

        Parameters
        ----------
        :param PWM_estimators: PWM_estimators object, optional
            PWM_estimators object containing PWM estimators for initializing parameters.
        :param initParams: str or numpy.ndarray, optional
            Initial parameters for ML estimation. 'auto' to calculate automatically using PWM estimators.
        :param option: int, optional
            Option for ML estimation.
        :param estimate_pi: bool, optional
            Whether to estimate π parameter or not.
        """
        if initParams == 'auto' and PWM_estimators==None:
            raise ValueError('Automatic calculation of initParams needs PWM_estimators!')
        elif initParams == 'auto':
            initParams = automatic_parameter_initialization(PWM_estimators.values, 
                                                            self.TimeSeries.corr, 
                                                            ts=self.TimeSeries.ts)
        for i, ho_stat in enumerate(self.high_order_stats):
            def cost(params):
                gamma, mu, sigma, pi = params
                cst = - log_likelihoods(ho_stat, gamma=gamma, mu=mu, sigma=sigma, pi=pi, option=option, ts=self.TimeSeries.ts, corr=self.TimeSeries.corr)
                return cst

            results = misc.minimize(cost, initParams[i], method='Nelder-Mead')
    
            gamma, mu, sigma, pi = results.x
            
            self.values.append([gamma, mu, sigma, pi])
        self.values = np.array(self.values)
    
    def get_statistics(self, gamma_true):
        r"""
        Compute statistics of the ML estimators using a true :math:`\gamma` value.

        Parameters
        ----------
        :param gamma_true: float
            True :math:`\gamma` value for calculating statistics.
        """
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
    r"""
    TimeSeries class for simulating and analyzing time series data.

    Parameters
    ----------
    n : int
        Length of the time series.
    distr : str, optional
        Distribution to draw from. Default is 'GEV'.
    correlation : str, optional
        Correlation type, choose from ['IID', 'ARMAX', 'AR']. Default is 'IID'.
    modelparams : list, optional
        Parameters belonging to the specified distribution. Default is [0].
    ts : float, optional
        Time series parameter alpha in [0,1]. Default is 0.
        
    Attributes
    ----------
    values : list
        List to store simulated time series.
    distr : str
        Distribution type.
    corr : str
        Correlation type.
    modelparams : list
        Model parameters.
    ts : float
        Time series parameter.
    len : int
        Length of the time series.
    blockmaxima : list
        List to store block maxima.
    high_order_stats : list
        List to store high order statistics.

    Methods
    -------
    simulate(rep=10, seeds='default'):
        Simulate time series data.
    get_blockmaxima(block_size=2, stride='DBM', rep=10):
        Extract block maxima from simulated data.
    get_HOS(orderstats=2, block_size=2, stride='DBM', rep=10):
        Extract high order statistics from simulated data.

     Example
    -------
    >>> # Create a TimeSeries object
    >>> ts = TimeSeries(n=100, distr='GEV', correlation='ARMAX', modelparams=[0.5], ts=0.6)
    >>> 
    >>> # Simulate time series data
    >>> ts.simulate(rep=5, seeds=[42, 123, 456, 789, 1011])
    >>> 
    >>> # Extract block maxima
    >>> ts.get_blockmaxima(block_size=5, stride='DBM', rep=5)
    >>> 
    >>> # Extract high order statistics
    >>> ts.get_HOS(orderstats=3, block_size=5, stride='DBM', rep=5)
    """
    
    def __init__(self, n, distr='GEV', correlation='IID', modelparams=[0], ts=0):
        self.values = []
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
        r"""
        Simulate time series data.

        Parameters
        ----------
        rep : int, optional
            Number of repetitions for simulation. Default is 10.
        seeds : {str, list}, optional
            Seed(s) for random number generation. Default is 'default'.

        Raises
        ------
        ValueError
            If the number of given seeds does not match the number of repetitions.
        """
        # ensure to overwrite existing
        self.values = []
        self.reps = 10
        if seeds == 'default':
            for i in range(rep):
                series = misc.simulate_timeseries(self.len,
                                             distr=self.distr, 
                                             correlation=self.corr, 
                                             modelparams=self.modelparams, 
                                             ts=self.ts, 
                                             seed=i)
                self.values.append(series)
        elif seeds == None:
            for i in range(rep):
                series = misc.simulate_timeseries(self.len,
                                             distr=self.distr, 
                                             correlation=self.corr, 
                                             modelparams=self.modelparams, 
                                             ts=self.ts, 
                                             seed=None)
                self.values.append(series)
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
                    self.values.append(series)
        
    def get_blockmaxima(self, block_size=2, stride='DBM', rep=10):
        r"""
        Extract block maxima from simulated data.

        Parameters
        ----------
        block_size : int, optional
            Size of blocks for maxima extraction. Default is 2.
        stride : str, optional
            Type of stride, choose from ['SBM', 'DBM']. Default is 'DBM'.
        rep : int, optional
            Number of repetitions for extraction. Default is 10.
        """
        # ensure to overwrite existing
        self.blockmaxima = []
        if self.values == []:
            warnings.warn('TimeSeries was not simulated yet, I will do it for you. For more flexibility, simulate first!')
            self.simulate(rep=rep)
        self.block_size = block_size
        self.stride = stride
        for series in self.values:
            bms = extract_BM(series, block_size=block_size, stride=stride)
            self.blockmaxima.append(bms)
    

    def get_HOS(self, orderstats = 2, block_size=2, stride='DBM', rep=10):
        r"""
        Extract high order statistics from simulated data.

        Parameters
        ----------
        orderstats : int, optional
            Order of statistics. Default is 2.
        block_size : int, optional
            Size of blocks for statistics extraction. Default is 2.
        stride : str, optional
            Type of stride, choose from ['SBM', 'DBM']. Default is 'DBM'.
        rep : int, optional
            Number of repetitions for extraction. Default is 10.
        """
        # ensure to overwrite existing
        self.high_order_stats = []
        if self.values == []:
            warnings.warn('TimeSeries was not simulated yet, I will do it for you. For more flexibility, simulate first!')
            self.simulate(rep=rep)
        self.block_size = block_size
        self.stride = stride
        for series in self.values:
            hos = extract_HOS(series, orderstats=orderstats, block_size=block_size, stride=stride)
            self.high_order_stats.append(hos)

class HighOrderStats:
    r"""HighOrderStats class for calculating and analyzing high-order statistics of time series data.

    Notes
    -----
    This class provides functionality for calculating Probability Weighted Moment (PWM) estimators and
    Maximum Likelihood (ML) estimators from a given TimeSeries object.

    Methods
    -------
    get_PWM_estimation():
        Calculate the PWM estimators for the time series data.
    
    get_ML_estimation(initParams='auto', option=1, estimate_pi=False):
        Calculate the ML estimators for the time series data.

    Attributes
    ----------
    TimeSeries : TimeSeries
        The TimeSeries object containing the time series data.

    high_order_stats : list
        List of high-order statistics extracted from the TimeSeries object.

    blockmaxima : list
        List of block maxima derived from the high-order statistics.

    gamma_true : float
        True gamma parameter of the GEV distribution derived from the TimeSeries object.

    PWM_estimators : PWM_estimators
        Instance of PWM_estimators class for calculating PWM estimators.

    ML_estimators : ML_estimators
        Instance of ML_estimators class for calculating ML estimators.
    
    Example
    -------
    >>> # Create a TimeSeries object
    >>> ts = TimeSeries(n=100, distr='GEV', correlation='ARMAX', modelparams=[0.5], ts=0.6)
    >>> 
    >>> # Simulate time series data
    >>> ts.simulate(rep=5, seeds=[42, 123, 456, 789, 1011])
    >>> 
    >>> # Initialize HighOrderStats object
    >>> hos = HighOrderStats(ts)
    >>> 
    >>> # Calculate PWM estimators
    >>> hos.get_PWM_estimation()
    >>> 
    >>> # Calculate ML estimators
    >>> hos.get_ML_estimation(initParams='auto', option=1, estimate_pi=False)
    """
    def __init__(self, TimeSeries):
        r"""
        Initialize the HighOrderStats object.

        Parameters
        ----------
        TimeSeries : TimeSeries
            TimeSeries object containing the time series data.

        """
        self.TimeSeries = TimeSeries
        if TimeSeries.high_order_stats == []:
            warnings.warn('TimeSeries does not have high_order_statistics, I will do it for you. For more flexibility, compute them first!')
        self.high_order_stats = TimeSeries.high_order_stats
        self.blockmaxima = [ho_stat.T[-1] for ho_stat in self.high_order_stats]
        self.gamma_true = misc.modelparams2gamma_true(TimeSeries.distr, TimeSeries.corr, TimeSeries.modelparams)
        self.PWM_estimators = PWM_estimators(TimeSeries=self.TimeSeries)
        self.ML_estimators = ML_estimators(TimeSeries=self.TimeSeries)
    
    def get_PWM_estimation(self):
        r"""
        Calculate the Probability Weighted Moment (PWM) estimators.
        """
        # ensure to overwrite existing
        self.PWM_estimators = PWM_estimators(TimeSeries=self.TimeSeries)
        self.PWM_estimators.get_PWM_estimation()
        self.PWM_estimators.get_statistics(gamma_true=self.gamma_true)
        
    def get_ML_estimation(self, initParams = 'auto', option=1, estimate_pi=False):
        r"""
        Calculate the Maximum Likelihood (ML) estimators.

        Parameters
        ----------
        initParams : str or array-like, optional
            Method for initializing parameters. Default is 'auto', which uses automatic parameter initialization.
        option : int, optional
            Option for ML estimation. Default is 1.
        estimate_pi : bool, optional
            Whether to estimate the pi parameter. Default is False.
        """
        # ensure to overwrite existing
        self.ML_estimators = ML_estimators(TimeSeries=self.TimeSeries)
        if self.PWM_estimators.values == [] and initParams == 'auto':
            self.get_PWM_estimation()
        self.ML_estimators.get_ML_estimation(PWM_estimators=self.PWM_estimators, initParams = initParams, option=option, estimate_pi=estimate_pi)
        self.ML_estimators.get_statistics(gamma_true=self.gamma_true)
        

