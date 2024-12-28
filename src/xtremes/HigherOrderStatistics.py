import numpy as np
from tqdm import tqdm
import scipy
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import genextreme, invweibull, weibull_max, gumbel_r
from scipy.special import gamma as Gamma
from pynverse import inversefunc
# import asyncio
import pickle
import warnings
import xtremes.miscellaneous as misc
import matplotlib.pyplot as plt

############################### FUNCTIONS ###############################

# log-likelihood

def log_likelihood(high_order_statistics,  gamma=0, mu=0, sigma=1, r=None):
    r"""
    Calculate the GEV log likelihood based on the two highest order statistics in three different ways.

    Parameters
    ----------
    high_order_statistics : numpy.ndarray
        A 2D array where each row contains the two highest order statistics for each observation.
    gamma : float, optional
        The shape parameter (γ) for the Generalized Extreme Value (GEV) distribution. Default is 0.
    mu : float, optional
        The location parameter (μ) for the GEV distribution. Default is 0.
    sigma : float, optional
        The scale parameter (σ) for the GEV distribution. Must be positive. Default is 1.
    r : int, optional
        The number of order statistics to calculate the log-likelihood on. If not specified, it uses all provided statistics.

    Returns
    -------
    float
        The calculated log likelihood.

    Notes
    -----
    - This function computes the log likelihood using the two highest order statistics and supports both the classical 
      Gumbel case (γ = 0) and the generalized case (γ ≠ 0).
    - The `high_order_statistics` array should be structured with the two highest order statistics per observation as rows.
    - The shape parameter γ controls the tail behavior of the distribution. When γ = 0, the distribution becomes the Gumbel type.
    - The `r` parameter controls how many order statistics are used for the likelihood calculation, typically `r=2` for two order statistics.

    Example
    -------
    >>> hos = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.5], [0.4, 0.6]])
    >>> log_likelihood(hos, gamma=0.5, mu=0, sigma=2, r=2)
    -7.494890426732856
    """
    if r == None:
        r = high_order_statistics.shape[1]
    
    unique_hos, counts = np.unique(high_order_statistics, axis=0, return_counts=True)
    # split into largest and second largest
    #maxima = unique_hos.T[1]
    #second = unique_hos.T[0]
    sigma = np.abs(sigma)
    if sigma < 0.01:
        # ensure no division by 0
        sigma += 0.01
    # standardize both

    #y_max = (maxima-mu)/sigma 
    #y_sec = (second-mu)/sigma
    unique_hos = (unique_hos.T - mu)/sigma
    out = np.zeros_like(unique_hos[0])

    if np.abs(gamma) < 0.0001:
        out = - r * np.log(sigma) - np.exp(-unique_hos[-r]) - np.sum(unique_hos[-r:,], axis=0)
    else:
        mask = np.bitwise_and((1+gamma*unique_hos[-1]>0), (1+gamma*unique_hos[0]>0)) # ensure all conditions fulfilled
        f = lambda x: np.log(1+gamma*x[mask])*(1/gamma+1)
        out[mask] = -r * np.log(sigma) - (1+gamma*unique_hos[-r][mask])**(-1/gamma) - np.sum(np.apply_along_axis(f, 1, unique_hos[-r:,]), axis=0)
        out[np.invert(mask)] = -1000 # ln(0)
    
    joint_ll = np.dot(out, counts)
    
    return joint_ll

def Frechet_log_likelihood(high_order_statistics, alpha=1, sigma=1, r=None):
    r"""
    Calculate the 2-parameter Frechet log likelihood based on the highest order statistics. 
    The calculation can be done using either the joint likelihood of the top two order statistics or the product of their marginals.

    Parameters
    ----------
    high_order_statistics : numpy.ndarray
        A 2D array where each row contains the two highest order statistics for each observation.
    alpha : float, optional
        The shape parameter (α) for the Frechet distribution. Default is 1. Controls the tail behavior of the distribution.
    sigma : float, optional
        The scale parameter (σ) for the Frechet distribution. Must be positive. Default is 1.
    r : int, optional
        The number of order statistics to calculate the log-likelihood on. If not specified, it uses all provided statistics.

    Returns
    -------
    float
        The calculated log likelihood for the given data under the Frechet distribution.

    Notes
    -----
    - This function computes the log likelihood using the two highest order statistics from each observation, with a focus on the Frechet distribution.
    - The shape parameter `alpha` determines the heaviness of the tail in the distribution, and the scale parameter `sigma` must be strictly positive.
    - The `high_order_statistics` array should be structured such that each row represents the two highest order statistics for an observation.
    - The function can either calculate the joint likelihood of the top two order statistics or consider the product of their marginals, depending on the values used.

    Example
    -------
    >>> hos = np.array([[0.5, 1.0], [1.5, 2.0], [1.2, 2.2], [2.0, 3.0]])
    >>> Frechet_log_likelihood(hos, alpha=2, sigma=1.5, r=2)
    -15.78467219003245
    """

    if r == None or r>high_order_statistics.shape[1]:
        r = high_order_statistics.shape[1]
    unique_hos, counts = np.unique(high_order_statistics, axis=0, return_counts=True)
    sigma = np.abs(sigma)
    unique_hos = unique_hos.T 
    if sigma < 0.01:
        # ensure no division by 0
        sigma += 0.01
    out = np.zeros_like(unique_hos[0])

    if alpha < 0.00001:
        out = -np.inf * np.ones_like(unique_hos[-r])
    else:
        mask = (unique_hos[-r]>0)
        f = lambda x: np.log(x[mask]) 
        out[mask] = r * np.log(alpha) + r * alpha * np.log(sigma) - (alpha+1)*np.sum(np.apply_along_axis(f, 1, unique_hos[-r:,]), axis=0) - (unique_hos[-r][mask]/sigma)**(-alpha)
        # out[mask] = - r * np.log(alpha) + r * 1/alpha * np.log(sigma) - (1/alpha+1)*np.sum(np.apply_along_axis(f, 1, unique_hos[-r:,]), axis=0) - (unique_hos[-r][mask]/sigma)**(-1/alpha)
        out[np.invert(mask)] = - 1e6 # np.inf # ln(0)
    joint_ll = np.dot(out, counts)
    
    return joint_ll



def extract_BM(timeseries, block_size=10, stride='DBM', return_indices=False):
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
    if return_indices:
        indices = [np.argmax(timeseries[i*p:i*p+r])+i*p for i in range((n-r)//p+1)]
        return np.array(indices), np.array(out)
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
    return  np.array(PWM_estimators)
    #initParams = np.array(PWM_estimators)
    ## pi parameter from theory
    #if corr == 'ARMAX':
    #    pis = np.ones(shape=(1,len(PWM_estimators))) *(0.5)#* misc.invsigmoid(1-ts)
    #elif corr == 'IID':
    #    pis = np.ones(shape=(1,len(PWM_estimators)))
    #else:
    #    pis = np.ones(shape=(1,len(PWM_estimators)))
    #return np.append(initParams.T, pis, axis=0).T



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
    :param r: int, optional
        Number of orderstatistics to calculate the log-likelihood on. If not specified, use all provided
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
    :param r: int, optional
        Number of orderstatistics to calculate the log-likelihood on. If not specified, use all provided
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
    

# ############################### PWM & MLE ESTIMATOR CLASSES ###############################


class PWM_estimators:
    r"""
    Calculates Probability Weighted Moment (PWM) estimators from block maxima and computes statistics and confidence intervals for the GEV parameters.

    Notes
    -----
    This class provides methods to compute the Probability Weighted Moment (PWM) estimators and convert them into Generalized Extreme Value (GEV) 
    distribution parameters (gamma, mu, sigma). It also allows users to compute confidence intervals for the estimated parameters and assess 
    the statistical properties (mean, variance, bias, mean squared error) of the estimators compared to a true gamma value. Visualization options 
    are provided to plot the estimators and confidence intervals.

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
        Dictionary containing statistics (mean, variance, bias, mse) and confidence intervals of the PWM estimators.

    Methods
    -------
    :method get_PWM_estimation():
        Compute PWM estimators for each block maxima series and convert them into GEV parameters.
    :method get_statistics(gamma_true):
        Compute statistics of the PWM estimators using a true gamma value.
    :method get_CIs(alpha=0.05, method='symmetric'):
        Compute confidence intervals (CIs) for the GEV parameters using either symmetric quantiles or minimal width intervals.
    :method plot(param='gamma', show_CI=True, show_true=True, filename=None):
        Plot the PWM estimators for the GEV parameters (gamma, mu, sigma) with options to display confidence intervals and save the plot as an image.

    Raises
    ------
    :raises ValueError: 
        If block maxima or high order statistics are not available in the TimeSeries object.

    Examples
    --------
    >>> from TimeSeries import TimeSeries
    >>> from PWM_estimators import PWM_estimators
    >>> ts = TimeSeries(data)  # Initialize TimeSeries object with data
    >>> ts.get_blockmaxima(block_size=10, stride='SBM')  # Extract block maxima
    >>> pwm = PWM_estimators(ts)  # Initialize PWM_estimators object
    >>> pwm.get_PWM_estimation()  # Compute PWM estimators
    >>> pwm.get_statistics(0.1)  # Compute statistics with true gamma value 0.1
    >>> pwm.get_CIs(alpha=0.05, method='minimal_width')  # Compute confidence intervals
    >>> pwm.plot(param='gamma', show_CI=True, show_true=True, filename='PWM_plot.png')  # Plot the results and save as image
    >>> print(pwm.statistics)  # Print computed statistics
    {'gamma_mean': 0.25, 'gamma_variance': 0.005, 'gamma_bias': 0.02, 'gamma_mse': 0.01,
     'mu_mean': 1.15, 'mu_variance': 0.04, 'sigma_mean': 0.9, 'sigma_variance': 0.02,
     'gamma_CI': (0.2, 0.5), 'mu_CI': (1.1, 1.5), 'sigma_CI': (0.8, 1.0)}
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
        Compute Probability-Weighted Moment (PWM) estimators and convert them into GEV parameters.

        Notes
        -----
        This function iterates over each block maxima series, computes the Probability-Weighted Moments (PWMs), and converts the PWMs into the 
        Generalized Extreme Value (GEV) distribution parameters: shape (gamma), location (mu), and scale (sigma). The function utilizes the 
        `misc.PWM_estimation` function to calculate the PWMs and then applies `misc.PWM2GEV` to convert these moments into GEV parameters.

        The results for each block maxima series are stored in the `self.values` attribute, which is a NumPy array where each row corresponds to 
        the GEV parameters [gamma, mu, sigma] for a specific block maxima series.

        This function clears any previously stored values in `self.values` before appending new estimates.

        Parameters
        ----------
        None

        Example
        -------
        >>> estimator = PWM_estimators(timeseries_data)
        >>> estimator.get_PWM_estimation()
        >>> print(estimator.values)
        array([[0.2, 1.1, 0.8],
            [0.3, 1.2, 0.9]])

        Returns
        -------
        None
            The results are stored in `self.values`, a NumPy array containing the estimated GEV parameters for each block maxima series.
        """

        # clear values
        self.values = []
        for bms in self.blockmaxima:
            b0, b1, b2 = misc.PWM_estimation(bms) 
            gamma, mu, sigma = misc.PWM2GEV(b0, b1, b2)
            self.values.append([gamma, mu, sigma])
        self.values = np.array(self.values)
    
    def get_statistics(self, gamma_true):    
        r"""
        Compute statistics of the PWM estimators using the true gamma value.

        Notes
        -----
        This function calculates various statistical measures (mean, variance, bias, and mean squared error) for the estimated 
        Generalized Extreme Value (GEV) shape parameter (gamma) compared to a provided true value (`gamma_true`). It also computes 
        the mean and variance for the location (mu) and scale (sigma) parameters across all block maxima series.

        - **Mean Squared Error (MSE)**: Measures the average of the squares of the differences between the estimated and true gamma values.
        - **Bias**: Represents the systematic deviation of the estimated gamma values from the true gamma value.
        - **Variance**: Describes the spread of the estimated gamma values around their mean.

        If only one block maxima series is available, a warning is raised since variance cannot be computed.

        The computed statistics are stored in the `self.statistics` dictionary with the following keys:
        - 'gamma_mean', 'gamma_variance', 'gamma_bias', 'gamma_mse'
        - 'mu_mean', 'mu_variance'
        - 'sigma_mean', 'sigma_variance'

        Parameters
        ----------
        :param gamma_true: float
            The true value of the GEV shape parameter (gamma) used to compute bias and MSE.

        Example
        -------
        >>> estimator = PWM_estimators(timeseries_data)
        >>> estimator.get_PWM_estimation()
        >>> estimator.get_statistics(gamma_true=0.2)
        >>> print(estimator.statistics)
        {'gamma_mean': 0.25, 'gamma_variance': 0.005, 'gamma_bias': 0.02, 'gamma_mse': 0.01,
        'mu_mean': 1.15, 'mu_variance': 0.04, 'sigma_mean': 0.9, 'sigma_variance': 0.02}

        Returns
        -------
        None
            The results are stored in `self.statistics`, containing the calculated statistics for gamma, mu, and sigma.
        """

        gammas = self.values.T[0]
        mus = self.values.T[1]
        sigmas = self.values.T[2]
        if len(gammas) > 1:
            # gamma-related statistics
            MSE = sum((np.array(gammas) - gamma_true)**2)/(len(np.array(gammas))-1)
            mean = np.mean(gammas)
            variance = sum((np.array(gammas) - mean)**2)/(len(np.array(gammas))-1)
            bias = np.sqrt(MSE - variance)
            self.statistics['gamma_mean'] = mean
            self.statistics['gamma_variance'] = variance
            self.statistics['gamma_bias'] = bias
            self.statistics['gamma_mse'] = MSE
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
    
    def get_CIs(self, alpha=0.05, method = 'symmetric'):
        r"""
        Compute confidence intervals (CIs) for the GEV parameters using different methods.

        Notes
        -----
        This function calculates confidence intervals (CIs) for the Generalized Extreme Value (GEV) parameters (gamma, mu, and sigma) estimated from 
        Probability-Weighted Moments (PWM). The user can choose between two methods for computing the confidence intervals:
        
        - **'symmetric'**: This method uses the quantiles of the distribution of parameter estimates to compute symmetric confidence intervals.
        - **'minimal_width'**: This method finds the interval with minimal width that contains the desired proportion (1 - alpha) of the sorted parameter estimates.

        For each block maxima series, confidence intervals for the GEV shape (gamma), location (mu), and scale (sigma) parameters are calculated. The results 
        are stored in the `self.statistics` dictionary, with keys 'gamma_CI', 'mu_CI', and 'sigma_CI' corresponding to the computed confidence intervals.

        Parameters
        ----------
        :param alpha: float, optional
            Significance level for the confidence intervals (default is 0.05, for a 95% CI).
        :param method: str, optional
            Method for computing the confidence intervals. Options are:
            - 'symmetric': Uses quantiles to compute symmetric CIs (default).
            - 'minimal_width': Computes the minimal width interval containing (1 - alpha) of the estimates.
            
        Example
        -------
        >>> estimator = PWM_Estimators(timeseries_data)
        >>> estimator.get_PWM_estimation()
        >>> estimator.get_CIs(alpha=0.05, method='minimal_width')
        >>> print(estimator.statistics)
        {'gamma_CI': (0.2, 0.5), 'mu_CI': (1.1, 1.5), 'sigma_CI': (0.8, 1.0)}

        Returns
        -------
        None
            The results are stored in `self.statistics`, which contains the confidence intervals for each GEV parameter.
         """

        if method == 'symmetric':
            lower = np.quantile(self.values, alpha/2, axis=0)
            upper = np.quantile(self.values, (1-alpha/2), axis=0)
            CIs = np.stack([lower, upper], axis=1)
            for param, CI in zip(['gamma', 'mu', 'sigma'], CIs):
                self.statistics[param+'_CI'] = CI
        if method == 'minimal_width':
            sorted_values = np.sort(self.values, axis=0)
            n = len(sorted_values)
            best_interval = np.zeros((2, sorted_values.shape[1]))           
            for param_idx, param in zip(range(sorted_values.shape[1]),['gamma', 'mu', 'sigma']):
                param_min_width = np.inf
                param_best_interval = (None, None)
                
                for i in range(n):
                    j = int(np.floor((1 - alpha) * n)) + i
                    if j >= n:
                        break
                    width = sorted_values[j, param_idx] - sorted_values[i, param_idx]
                    if width < param_min_width:
                        param_min_width = width
                        param_best_interval = (sorted_values[i, param_idx], sorted_values[j, param_idx])
                
                self.statistics[param+'_CI'] = param_best_interval
    
    def plot(self, param='gamma', show_CI=True, show_true=True, filename=None):
        r"""
        Plot the PWM estimators and confidence intervals for the GEV parameters.

        Notes
        -----
        This function generates a plot showing the Probability-Weighted Moment (PWM) estimators for the Generalized Extreme Value (GEV) parameters 
        (gamma, mu, sigma) computed from block maxima. The user can choose to display confidence intervals (CIs) for each parameter.

        The plot is saved as a PNG image if the `save` parameter is set to True.

        Parameters
        ----------
        :param param: str, optional
            GEV parameter to plot (default is 'gamma').
        :param show_CI: bool, optional
            Flag indicating whether to display confidence intervals (default is True).
        :param show: bool, optional
            Flag indicating whether to display the plot.
        :param save: bool, optional
            Flag indicating whether to save the plot as a PNG image (default is False).
        :param filename: str, optional
            Name of the PNG file to save the plot (default is None).

        Example
        -------
        >>> estimator = PWM_estimators(timeseries_data)
        >>> estimator.get_PWM_estimation()
        >>> estimator.get_CIs(alpha=0.05, method='minimal_width')
        >>> estimator.plot(show_CI=True, show_true=True, save=True, filename='PWM_estimation.png')

        Returns
        -------
        None
            The plot is displayed in the console and saved as a PNG image if the `save` parameter is set to True.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title(f'PWM Estimation for {param}')
        ax.set_xlabel('Estimation for '+param)
        ax.set_ylabel('Frequency')
        idx = ['gamma', 'mu', 'sigma'].index(param)
        ax.hist(self.values.T[idx], bins=20, color='skyblue', edgecolor='black', alpha=0.7,label='Estimation')
        if show_CI:
            CI = self.statistics[param+'_CI']
            ax.axvline(CI[0], color='red', linestyle='dashed', linewidth=2, label='CI lower bound')
            ax.axvline(CI[1], color='red', linestyle='dashed', linewidth=2, label='CI upper bound') 
        plt.legend()
        if filename:
            plt.savefig(filename)
        if show_true:
            plt.show()


                

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
    >>> from hos import TimeSeries, ML_estimators, PWM_estimators
    >>> blockmaxima_data = np.random.normal(loc=10, scale=2, size=100)
    >>> high_order_stats_data = np.random.normal(loc=5, scale=1, size=(100, 3))
    >>> ts = TimeSeries(blockmaxima=blockmaxima_data, high_order_stats=high_order_stats_data, corr='IID', ts=0.5)
    >>> pwm = PWM_estimators(ts)
    >>> pwm.get_PWM_estimation()
    >>> ml = ML_estimators(ts)
    >>> ml.get_ML_estimation(PWM_estimators=pwm)
    >>> ml.get_statistics(gamma_true=0.1)
    >>> print("ML Estimators:")
    >>> print(ml.values)
    >>> print("\nStatistics:")
    >>> print(ml.statistics)
    """
    def __init__(self, TimeSeries):
        # TimeSeries object needs high order statistics
        if TimeSeries.high_order_stats == []:
            if TimeSeries.blockmaxima != []:
                TimeSeries.get_HOS(orderstats = 1, block_size=TimeSeries.block_size, stride=TimeSeries.stride)
                self.high_order_stats = TimeSeries.high_order_stats
            else:
                raise ValueError('Caluclate high order statistics or block maxima first!')
        else:
            self.high_order_stats = TimeSeries.high_order_stats
        self.TimeSeries = TimeSeries
        self.values = []
        self.statistics = {}
    
    def __len__(self):
        return len(self.values)

    def get_ML_estimation(self, PWM_estimators=None, initParams = 'auto', r=None):
        r"""
        Compute Maximum Likelihood (ML) estimators for each series of high order statistics within the `ML_estimators` class.

        This method fits the Generalized Extreme Value (GEV) distribution to each series of high order statistics
        using Maximum Likelihood Estimation (MLE) by optimizing the log-likelihood function.

        Parameters
        ----------
        PWM_estimators : PWM_estimators object, optional
            An object containing PWM (Probability Weighted Moments) estimators. This is used for initializing the 
            parameters in the ML estimation if `initParams` is set to 'auto'. Required if `initParams` is 'auto'.
        initParams : str or numpy.ndarray, optional
            Initial parameters for the ML estimation. If 'auto', the initial parameters will be computed automatically 
            using the `PWM_estimators` object. If a NumPy array is provided, these will be used as initial parameter values.
            Default is 'auto'.
        r : int, optional
            The number of order statistics to calculate the log-likelihood on. If not specified, all provided 
            order statistics will be used.

        Returns
        -------
        None
            This method updates the `self.values` attribute of the `ML_estimators` object with the estimated parameters 
            (gamma, mu, sigma) for each series of high order statistics.

        Raises
        ------
        ValueError
            If `initParams` is set to 'auto' and no `PWM_estimators` are provided, a ValueError is raised.

        Notes
        -----
        - This method performs Maximum Likelihood Estimation (MLE) to fit the Generalized Extreme Value (GEV) distribution to the high order statistics within the `ML_estimators` class.
        - The method uses optimization techniques such as Nelder-Mead (and optionally COBYLA) to minimize the negative log-likelihood.
        - If `initParams` is set to 'auto', the initial parameters for the optimization are derived using the `PWM_estimators` object.
        - The optimization results (gamma, mu, sigma) are stored in the `self.values` list for each series of high order statistics.
        
        """

        self.values = []
        if initParams == 'auto' and PWM_estimators==None:
            raise ValueError('Automatic calculation of initParams needs PWM_estimators!')
        elif initParams == 'auto':
            initParams = automatic_parameter_initialization(PWM_estimators.values, 
                                                            self.TimeSeries.corr, 
                                                            ts=self.TimeSeries.ts)
        for i, ho_stat in enumerate(self.high_order_stats):
            def cost(params):
                gamma, mu, sigma = params
                cst = - log_likelihood(ho_stat, gamma=gamma, mu=mu, sigma=sigma, r=r)
                return cst
            
            # COBYLA und Nelder-Mead agree, in some versions cobyla does not want bounds
            results = misc.minimize(cost, initParams[i], method='Nelder-Mead', bounds=((-np.inf,np.inf),(-np.inf,np.inf),(1e-5,np.inf)))
            # results = misc.minimize(cost, initParams[i], method='COBYLA', bounds=((0,np.inf),(-np.inf,np.inf),(0,np.inf)))
            #constr1 = scipy.optimize.LinearConstraint(np.array([[1,0,0]]), lb=0, keep_feasible=False)
            #constr2 = scipy.optimize.LinearConstraint(np.array([[0,0,1]]), lb=0, keep_feasible=False)
            #results = misc.minimize(cost, initParams[i], method='COBYLA', constraints=[constr1, constr2])
    
            gamma, mu, sigma = results.x
            
            self.values.append([gamma, mu, sigma])
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
        if len(gammas) > 1:
            # gamma-related statistics
            MSE = sum((np.array(gammas) - gamma_true)**2)/(len(np.array(gammas))-1)
            mean = np.mean(gammas)
            variance = sum((np.array(gammas) - mean)**2)/(len(np.array(gammas))-1)
            bias = np.sqrt(MSE - variance)
            self.statistics['gamma_mean'] = mean
            self.statistics['gamma_variance'] = variance
            self.statistics['gamma_bias'] = bias
            self.statistics['gamma_mse'] = MSE
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

    def get_CIs(self, alpha=0.05, method = 'symmetric'):
        r"""
        Compute confidence intervals (CIs) for the GEV parameters using different methods.

        Notes
        -----
        This function calculates confidence intervals (CIs) for the Generalized Extreme Value (GEV) parameters (gamma, mu, and sigma) 
        estimated from Maximum Likelihood Estimators (MLE). The user can choose between two methods for computing the confidence intervals:
        
        - **'symmetric'**: This method uses the quantiles of the distribution of parameter estimates to compute symmetric confidence intervals.
        - **'minimal_width'**: This method finds the interval with minimal width that contains the desired proportion (1 - alpha) of the sorted parameter estimates.

        For each block maxima series, confidence intervals for the GEV shape (gamma), location (mu), and scale (sigma) parameters are calculated. The results 
        are stored in the `self.statistics` dictionary, with keys 'gamma_CI', 'mu_CI', and 'sigma_CI' corresponding to the computed confidence intervals.

        Parameters
        ----------
        :param alpha: float, optional
            Significance level for the confidence intervals (default is 0.05, for a 95% CI).
        :param method: str, optional
            Method for computing the confidence intervals. Options are:
            - 'symmetric': Uses quantiles to compute symmetric CIs (default).
            - 'minimal_width': Computes the minimal width interval containing (1 - alpha) of the estimates.
            
        Example
        -------
        >>> estimator = ML_stimator(timeseries_data)
        >>> estimator.get_PWM_estimation()
        >>> estimator.get_CIs(alpha=0.05, method='minimal_width')
        >>> print(estimator.statistics)
        {'gamma_CI': (0.2, 0.5), 'mu_CI': (1.1, 1.5), 'sigma_CI': (0.8, 1.0)}

        Returns
        -------
        None
            The results are stored in `self.statistics`, which contains the confidence intervals for each GEV parameter.
         """

        if method == 'symmetric':
            lower = np.quantile(self.values, alpha/2, axis=0)
            upper = np.quantile(self.values, (1-alpha/2), axis=0)
            CIs = np.stack([lower, upper], axis=1)
            for param, CI in zip(['gamma', 'mu', 'sigma'], CIs):
                self.statistics[param+'_CI'] = CI
        if method == 'minimal_width':
            sorted_values = np.sort(self.values, axis=0)
            n = len(sorted_values)
            best_interval = np.zeros((2, sorted_values.shape[1]))           
            for param_idx, param in zip(range(sorted_values.shape[1]),['gamma', 'mu', 'sigma']):
                param_min_width = np.inf
                param_best_interval = (None, None)
                
                for i in range(n):
                    j = int(np.floor((1 - alpha) * n)) + i
                    if j >= n:
                        break
                    width = sorted_values[j, param_idx] - sorted_values[i, param_idx]
                    if width < param_min_width:
                        param_min_width = width
                        param_best_interval = (sorted_values[i, param_idx], sorted_values[j, param_idx])
                
                self.statistics[param+'_CI'] = param_best_interval
    

    def plot(self, param='gamma', show_CI=True, show_true=True, filename=None):
        r"""
        Plot the ML estimators and confidence intervals for the GEV parameters.

        Notes
        -----
        This function generates a plot showing the Maximum Likelihood estimators for the Generalized Extreme Value (GEV) parameters 
        (gamma, mu, sigma) computed from block maxima. The user can choose to display confidence intervals (CIs) for each parameter

        The plot is saved as a PNG image if the `save` parameter is set to True.

        Parameters
        ----------
        :param param: str, optional
            GEV parameter to plot (default is 'gamma').
        :param show_CI: bool, optional
            Flag indicating whether to display confidence intervals (default is True).
        :param show: bool, optional
            Flag indicating whether to display the plot.
        :param save: bool, optional
            Flag indicating whether to save the plot as a PNG image (default is False).
        :param filename: str, optional
            Name of the PNG file to save the plot (default is None).

        Example
        -------
        >>> estimator = ML_estimators(timeseries_data)
        >>> estimator.get_ML_estimation()
        >>> estimator.get_CIs(alpha=0.05, method='minimal_width')
        >>> estimator.plot(show_CI=True, show_true=True, save=True, filename='PWM_estimation.png')

        Returns
        -------
        None
            The plot is displayed in the console and saved as a PNG image if the `save` parameter is set to True.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title(f'ML Estimation for {param}')
        ax.set_xlabel('Estimation for '+param)
        ax.set_ylabel('Frequency')
        idx = ['gamma', 'mu', 'sigma'].index(param)
        ax.hist(self.values.T[idx], bins=20, color='skyblue', edgecolor='black', alpha=0.7,label='Estimation')
        if show_CI:
            CI = self.statistics[param+'_CI']
            ax.axvline(CI[0], color='red', linestyle='dashed', linewidth=2, label='CI lower bound')
            ax.axvline(CI[1], color='red', linestyle='dashed', linewidth=2, label='CI upper bound') 
        plt.legend()
        if filename:
            plt.savefig(filename)
        if show_true:
            plt.show()
    

class Frechet_ML_estimators:
    r"""
    Maximum Likelihood Estimators (MLE) for Frechet parameters.

    This class calculates Maximum Likelihood Estimators (MLE) for the parameters of the 2-parameter Frechet 
    distribution using the method of maximum likelihood estimation on a series of high order statistics.

    Parameters
    ----------
    TimeSeries : TimeSeries
        The TimeSeries object containing the data (high order statistics) for which MLE estimators will be calculated.

    Attributes
    ----------
    values : numpy.ndarray
        An array containing the MLE estimators (alpha, sigma) for each set of high order statistics.
    statistics : dict
        A dictionary containing computed statistics from the MLE estimators such as mean, variance, bias, and MSE.

    Methods
    -------
    __len__()
        Returns the number of MLE estimators calculated.
    get_ML_estimation(PWM_estimators=None, initParams='auto', r=None)
        Computes the MLE estimators for the Frechet parameters (alpha, sigma).
    get_statistics(alpha_true)
        Computes statistics (mean, variance, bias, and MSE) of the MLE estimators using a true alpha value.

    Examples
    --------
    >>> import numpy as np
    >>> from hos import TimeSeries, Frechet_ML_estimators, PWM_estimators
    >>> blockmaxima_data = np.random.normal(loc=10, scale=2, size=100)
    >>> high_order_stats_data = np.random.normal(loc=5, scale=1, size=(100, 3))
    >>> ts = TimeSeries(blockmaxima=blockmaxima_data, high_order_stats=high_order_stats_data, corr='IID', ts=0.5)
    >>> pwm = PWM_estimators(ts)
    >>> pwm.get_PWM_estimation()
    >>> ml = Frechet_ML_estimators(ts)
    >>> ml.get_ML_estimation(PWM_estimators=pwm)
    >>> ml.get_statistics(alpha_true=0.1)
    >>> print("ML Estimators:")
    >>> print(ml.values)
    >>> print("\nStatistics:")
    >>> print(ml.statistics)
    """

    def __init__(self, TimeSeries):
        # TimeSeries object needs high order statistics
        if TimeSeries.high_order_stats == []:
            if TimeSeries.blockmaxima != []:
                TimeSeries.get_HOS(orderstats = 1, block_size=TimeSeries.block_size, stride=TimeSeries.stride)
                self.high_order_stats = TimeSeries.high_order_stats
            else:
                raise ValueError('Caluclate high order statistics or block maxima first!')
        else:
            self.high_order_stats = TimeSeries.high_order_stats
        self.TimeSeries = TimeSeries
        self.values = []
        self.statistics = {}
    
    def __len__(self):
        """
        Return the number of MLE estimators calculated.

        Returns
        -------
        int
            The number of sets of high order statistics for which MLE estimators have been calculated.
        """
        return len(self.values)

    def get_ML_estimation(self, PWM_estimators=None, initParams = 'auto', r=None):
        r"""
        Compute ML estimators (alpha, sigma) for each high order statistics series using Frechet distribution.

        Parameters
        ----------
        PWM_estimators : PWM_estimators object, optional
            PWM_estimators object containing PWM estimators for initializing parameters. Required if `initParams` 
            is set to 'auto'.
        initParams : str or numpy.ndarray, optional
            Initial parameters for ML estimation. 'auto' to calculate automatically using PWM estimators.
            If a numpy array is provided, these will be used as initial parameter values. Default is 'auto'.
        r : int, optional
            Number of order statistics to calculate the log-likelihood on. If not specified, all provided order 
            statistics will be used.

        Returns
        -------
        None
            Updates the `self.values` attribute with the estimated parameters (alpha, sigma) for each series 
            of high order statistics.

        Raises
        ------
        ValueError
            If `initParams` is set to 'auto' and no `PWM_estimators` are provided.
        """
        if initParams == 'auto' and PWM_estimators==None:
            raise ValueError('Automatic calculation of initParams needs PWM_estimators!')
        elif initParams == 'auto':
            initParams = automatic_parameter_initialization(PWM_estimators.values, 
                                                            self.TimeSeries.corr, 
                                                            ts=self.TimeSeries.ts)
        for i, ho_stat in enumerate(self.high_order_stats):
            def cost(params):
                alpha, sigma = params
                cst = - Frechet_log_likelihood(ho_stat, alpha=alpha, sigma=sigma, r=r)
                return cst
            
            inits = [1/initParams[i][0], initParams[i][2]]
            #its = initParams[i][::2]
            results = misc.minimize(cost, inits, method='Nelder-Mead', bounds=((1e-6, np.inf),(1e-5, np.inf)))
            # results = misc.minimize(cost, inits, method='COBYLA', bounds=((0, np.inf),(0, np.inf)))
            #constr1 = scipy.optimize.LinearConstraint(np.array([[1,0]]), lb=0, keep_feasible=False)
            #constr2 = scipy.optimize.LinearConstraint(np.array([[0,1]]), lb=0, keep_feasible=False)
            #results = misc.minimize(cost, initParams[i], method='COBYLA', constraints=[constr1, constr2])
    
            alpha, sigma = results.x
            
            self.values.append([alpha, sigma])
        self.values = np.array(self.values)

    def get_statistics(self, alpha_true):
        r"""
        Compute statistics (mean, variance, bias, and MSE) of the ML estimators using the true :math:`\alpha` value.

        Parameters
        ----------
        alpha_true : float
            The true value of :math:`\alpha` to calculate bias, MSE, and other statistics.

        Returns
        -------
        None
            Updates the `self.statistics` dictionary with calculated statistics such as mean, variance, bias, 
            and MSE for both :math:`\alpha` and :math:`\sigma`.

        Notes
        -----
        The statistics include:
        - Mean and variance for the estimated alpha and sigma values.
        - Bias and mean squared error (MSE) for the alpha estimates.
        """
        alphas = self.values.T[0]
        sigmas = self.values.T[1]
        if len(alphas) > 1:
            # gamma-related statistics
            alpha_MSE = sum((np.array(alphas) - alpha_true)**2)/(len(np.array(alphas))-1)
            alpha_mean = np.mean(alphas)
            alpha_variance = sum((np.array(alphas) - alpha_mean)**2)/(len(np.array(alphas))-1)
            alpha_bias = np.sqrt(alpha_MSE - alpha_variance)
            self.statistics['alpha_mean'] = alpha_mean
            self.statistics['alpha_variance'] = alpha_variance
            self.statistics['alpha_bias'] = alpha_bias
            self.statistics['alpha_mse'] = alpha_MSE
            # sigma-related statistics
            sigma_mean = np.mean(sigmas)
            sigma_variance = sum((np.array(sigmas) - sigma_mean)**2)/(len(np.array(sigmas))-1)
            self.statistics['sigma_mean'] = sigma_mean
            self.statistics['sigma_variance'] = sigma_variance

        else:
            warnings.warn('No variance can be computed on only one element!')
    
    def get_CIs(self, alpha=0.05, method = 'symmetric'):
        r"""
        Compute confidence intervals (CIs) for the Frechet parameters using different methods.

        Notes
        -----
        This function calculates confidence intervals (CIs) for the Frechet parameters (alpha and sigma) 
        estimated from Maximum Likelihood Estimators (MLE). The user can choose between two methods for computing the confidence intervals:
        
        - **'symmetric'**: This method uses the quantiles of the distribution of parameter estimates to compute symmetric confidence intervals.
        - **'minimal_width'**: This method finds the interval with minimal width that contains the desired proportion (1 - alpha) of the sorted parameter estimates.

        For each block maxima series, confidence intervals for the shape (alpha) and scale (sigma) parameters are calculated. The results 
        are stored in the `self.statistics` dictionary, with keys 'gamma_CI', 'mu_CI', and 'sigma_CI' corresponding to the computed confidence intervals.

        Do not get confused with alpha being the significance level as well as the shape parameter of the Frechet distribution.

        Parameters
        ----------
        :param alpha: float, optional
            Significance level for the confidence intervals (default is 0.05, for a 95% CI).
        :param method: str, optional
            Method for computing the confidence intervals. Options are:
            - 'symmetric': Uses quantiles to compute symmetric CIs (default).
            - 'minimal_width': Computes the minimal width interval containing (1 - alpha) of the estimates.
            
        Example
        -------
        >>> estimator = ML_stimator(timeseries_data)
        >>> estimator.get_PWM_estimation()
        >>> estimator.get_CIs(alpha=0.05, method='minimal_width')
        >>> print(estimator.statistics)


        Returns
        -------
        None
            The results are stored in `self.statistics`, which contains the confidence intervals for each GEV parameter.
         """

        if method == 'symmetric':
            lower = np.quantile(self.values, alpha/2, axis=0)
            upper = np.quantile(self.values, (1-alpha/2), axis=0)
            CIs = np.stack([lower, upper], axis=1)
            for param, CI in zip(['alpha', 'sigma'], CIs):
                self.statistics[param+'_CI'] = CI
        if method == 'minimal_width':
            sorted_values = np.sort(self.values, axis=0)
            n = len(sorted_values)
            best_interval = np.zeros((2, sorted_values.shape[1]))           
            for param_idx, param in zip(range(sorted_values.shape[1]),['alpha', 'sigma']):
                param_min_width = np.inf
                param_best_interval = (None, None)
                
                for i in range(n):
                    j = int(np.floor((1 - alpha) * n)) + i
                    if j >= n:
                        break
                    width = sorted_values[j, param_idx] - sorted_values[i, param_idx]
                    if width < param_min_width:
                        param_min_width = width
                        param_best_interval = (sorted_values[i, param_idx], sorted_values[j, param_idx])
                
                self.statistics[param+'_CI'] = param_best_interval
    
    def plot(self, param='alpha', show_CI=True, show_true=True, filename=None):
        r"""
        Plot the ML estimators and confidence intervals for the GEV parameters.

        Notes
        -----
        This function generates a plot showing the Maximum Likelihood estimators for the Generalized Extreme Value (GEV) parameters 
        (gamma, mu, sigma) computed from block maxima. The user can choose to display confidence intervals (CIs) for each parameter

        The plot is saved as a PNG image if the `save` parameter is set to True.

        Parameters
        ----------
        :param param: str, optional
            GEV parameter to plot (default is 'gamma').
        :param show_CI: bool, optional
            Flag indicating whether to display confidence intervals (default is True).
        :param show: bool, optional
            Flag indicating whether to display the plot.
        :param save: bool, optional
            Flag indicating whether to save the plot as a PNG image (default is False).
        :param filename: str, optional
            Name of the PNG file to save the plot (default is None).

        Example
        -------
        >>> estimator = PWM_estimators(timeseries_data)
        >>> estimator.get_PWM_estimation()
        >>> estimator.get_CIs(alpha=0.05, method='minimal_width')
        >>> estimator.plot(show_CI=True, show_true=True, save=True, filename='PWM_estimation.png')

        Returns
        -------
        None
            The plot is displayed in the console and saved as a PNG image if the `save` parameter is set to True.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title(f'ML Estimation for {param}')
        ax.set_xlabel('Estimation for '+param)
        ax.set_ylabel('Frequency')
        idx = ['alpha', 'sigma'].index(param)
        ax.hist(self.values.T[idx], bins=20, color='skyblue', edgecolor='black', alpha=0.7,label='Estimation')
        if show_CI:
            CI = self.statistics[param+'_CI']
            ax.axvline(CI[0], color='red', linestyle='dashed', linewidth=2, label='CI lower bound')
            ax.axvline(CI[1], color='red', linestyle='dashed', linewidth=2, label='CI upper bound') 
        plt.legend()
        if filename:
            plt.savefig(filename)
        if show_true:
            plt.show()

############################### TIMESERIES CLASS ###############################


class TimeSeries:
    r"""
    TimeSeries class for simulating and analyzing time series data with optional correlation structures.

    This class is designed to simulate time series data based on specified distributions and correlation types. 
    It also provides methods to extract block maxima and high order statistics from the simulated data.

    Parameters
    ----------
    n : int
        The length of the time series.
    distr : str, optional
        The distribution to simulate the time series data from. Default is 'GEV' (Generalized Extreme Value).
    correlation : str, optional
        The type of correlation structure. Options include ['IID', 'ARMAX', 'AR']. Default is 'IID' 
        (independent and identically distributed).
    modelparams : list, optional
        The parameters of the specified distribution. Default is [0].
    ts : float, optional
        A parameter for controlling the time series characteristics, particularly for correlated series (e.g., in AR models).
        Must be in the range [0, 1]. Default is 0.
        
    Attributes
    ----------
    values : list
        A list to store the simulated time series data.
    distr : str
        The type of distribution used for generating the time series data.
    corr : str
        The type of correlation structure applied to the time series.
    modelparams : list
        The parameters of the chosen distribution model.
    ts : float
        A parameter controlling the correlation or time series structure, if applicable.
    len : int
        The length of the time series.
    blockmaxima : list
        A list storing block maxima extracted from the simulated data.
    high_order_stats : list
        A list storing the high order statistics extracted from the simulated data.

    Methods
    -------
    simulate(rep=10, seeds='default'):
        Simulates time series data based on the given distribution and correlation type.
    get_blockmaxima(block_size=2, stride='DBM', rep=10):
        Extracts block maxima from the simulated time series data.
    get_HOS(orderstats=2, block_size=2, stride='DBM', rep=10):
        Extracts high order statistics from the simulated time series data.

    Examples
    --------
    >>> # Create a TimeSeries object
    >>> ts = TimeSeries(n=100, distr='GEV', correlation='ARMAX', modelparams=[0.5], ts=0.6)
    >>> # Simulate time series data with 5 repetitions and specific seeds
    >>> ts.simulate(rep=5, seeds=[42, 123, 456, 789, 1011])
    >>> # Extract block maxima using a block size of 5
    >>> ts.get_blockmaxima(block_size=5, stride='DBM', rep=5)
    >>> # Extract high order statistics (order 3) using the same block size and stride
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

        This method generates time series data based on the specified distribution, correlation type, and other parameters.

        Parameters
        ----------
        rep : int, optional
            The number of repetitions (simulations) to run. Default is 10.
        seeds : {str, list}, optional
            Seed(s) for random number generation. If 'default', a sequence of seeds is automatically generated 
            based on the number of repetitions. If a list of seeds is provided, it must match the number of 
            repetitions (`rep`). Default is 'default'.

        Raises
        ------
        ValueError
            If the number of provided seeds does not match the number of repetitions.
        """
        # ensure to overwrite existing
        self.values = []
        self.reps = rep
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
        Extract block maxima from simulated time series data.

        Block maxima are the maximum values extracted from blocks of the time series data.

        Parameters
        ----------
        block_size : int, optional
            The size of blocks from which to extract maxima. Default is 2.
        stride : {str, int}, optional
            The stride or step type used for block extraction. Choose from ['SBM' (Sliding Block Maxima), 'DBM' 
            (Disjoint Block Maxima)] or specify an integer as a step size. Default is 'DBM'.
        rep : int, optional
            The number of repetitions for maxima extraction. This should match the number of simulations. Default is 10.
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
        Extract high order statistics from simulated time series data.

        High order statistics include values such as the second-largest value within each block of the time series.

        Parameters
        ----------
        orderstats : int, optional
            The order of statistics to extract. Default is 2 (i.e., second-highest value).
        block_size : int, optional
            The size of blocks from which to extract the statistics. Default is 2.
        stride : str, optional
            The stride or step type used for block extraction. Choose from ['SBM', 'DBM']. Default is 'DBM'.
        rep : int, optional
            The number of repetitions for extracting high order statistics. Should match the number of simulations. Default is 10.
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
    
    def plot(self, rep=1, filename=None):
        r"""
        Plot the simulated time series data.

        This method generates a plot showing the simulated time series data. The user can choose to display 
        data from specific repetitions or all repetitions.

        Parameters
        ----------
        rep : int or list, optional
            The repetition number(s) to plot. If 0, all repetitions are plotted. If a list is provided, 
            only the specified repetitions are plotted. Default is 1.
        filename : str, optional
            The name of the PNG file to save the plot. If None, the plot is displayed but not saved. Default is None.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title('Simulated Time Series Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        
        if rep == 0:
            for i, series in enumerate(self.values):
                ax.plot(series, label=f'Rep {i+1}')
        elif isinstance(rep, list):
            for r in rep:
                ax.plot(self.values[r-1], label=f'Rep {r}')
        else:
            ax.plot(self.values[rep-1], label=f'Rep {rep}')
        
        plt.legend()
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_with_blockmaxima(self, rep=1, plotlim=300, filename=None):
        r"""
        Plot the simulated time series data along with block maxima.

        This method generates a plot showing the simulated time series data along with the block maxima. 
        The user can choose to display data from specific repetitions or all repetitions.

        Parameters
        ----------
        rep : int, optional
            The repetition number to plot. Default is 1.
        plotlim : int, optional
            The limit for the number of data points to plot. Default is 300.
        filename : str, optional
            The name of the PNG file to save the plot. If None, the plot is displayed but not saved. Default is None.
        """
        vals = self.values[rep-1]
        maxima = self.blockmaxima[rep-1]
        max_idx = np.where(np.in1d(vals, maxima))[0]  # find indices of maxima in timeseries

        fig = plt.figure(figsize=(16, 9))
        plt.tight_layout()
        plt.title('Time Series with Block Maxima')
        plt.plot(vals[:plotlim], label='Time Series')
        if self.stride == 'DBM':
            plt.scatter(max_idx[max_idx < plotlim], maxima[max_idx < plotlim], c='r', s=30, label='Block Maxima')
        if self.stride == 'SBM':  
            plt.scatter(np.arange(0,plotlim,1)+self.block_size/2, maxima[:plotlim], c='r',s=10, label='sliding maxima')
        for k in range((self.len + self.block_size) // self.block_size):
            if k < plotlim // self.block_size:
                plt.plot([self.block_size * k, self.block_size * k], [0, np.ceil(max(vals[:plotlim]))], c='k', lw=0.4)
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        
        if filename:
            plt.savefig(filename)
        plt.show()
    
    
    def plot_blockmaxima(self, rep=1, filename=None, plot_type='line'):
        r"""
        Plot the extracted block maxima.

        This method generates a plot showing the extracted block maxima from the simulated time series data. 
        The user can choose to display block maxima from specific repetitions or all repetitions. The plot 
        can be either a line plot or a histogram.

        Parameters
        ----------
        rep : int or list, optional
            The repetition number(s) to plot. If 0, all repetitions are plotted. If a list is provided, 
            only the specified repetitions are plotted. Default is 1.
        filename : str, optional
            The name of the PNG file to save the plot. If None, the plot is displayed but not saved. Default is None.
        plot_type : str, optional
            The type of plot to generate. Options are 'line' for a line plot and 'hist' for a histogram. Default is 'line'.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if plot_type == 'line':
            ax.set_title('Block Maxima (Line Plot)')
            ax.set_xlabel('Block')
            ax.set_ylabel('Value')
            
            if rep == 0:
                for i, bms in enumerate(self.blockmaxima):
                    ax.plot(bms, label=f'Rep {i+1}')
            elif isinstance(rep, list):
                for r in rep:
                    ax.plot(self.blockmaxima[r-1], label=f'Rep {r}')
            else:
                ax.plot(self.blockmaxima[rep-1], label=f'Rep {rep}')
            
            plt.legend()
        
        elif plot_type == 'hist':
            ax.set_title('Block Maxima (Histogram)')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
            if rep == 0:
                all_bms = np.concatenate(self.blockmaxima)
                ax.hist(all_bms, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            elif isinstance(rep, list):
                for r in rep:
                    ax.hist(self.blockmaxima[r-1], bins=20, alpha=0.7, label=f'Rep {r}')
                plt.legend()
            else:
                ax.hist(self.blockmaxima[rep-1], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        
        if filename:
            plt.savefig(filename)
        plt.show()


############################### HIGHORDERDSTATS CLASS ###############################


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
    
    get_ML_estimation(initParams='auto', FrechetOrGEV = 'Frechet', option=1, estimate_pi=False):
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
    >>> # Simulate time series data
    >>> ts.simulate(rep=5, seeds=[42, 123, 456, 789, 1011])
    >>> # Initialize HighOrderStats object
    >>> hos = HighOrderStats(ts)
    >>> # Calculate PWM estimators
    >>> hos.get_PWM_estimation()
    >>> # Calculate ML estimators
    >>> hos.get_ML_estimation(initParams='auto', r=1)
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
        
    def get_ML_estimation(self, initParams = 'auto', r=None, FrechetOrGEV = 'Frechet'):
        r"""
        Calculate the Maximum Likelihood (ML) estimators.

        Parameters
        ----------
        initParams : str or array-like, optional
            Method for initializing parameters. Default is 'auto', which uses automatic parameter initialization.
        
        r : int, optional
            Number of order statistics to calculate the log-likelihood on. If not specified, use all provided.
        
        FrechetOrGEV : str, optional
            Whether to fit the Frechet or GEV distribution.

        Notes
        -----
        This function performs maximum likelihood estimation based on either the Frechet or GEV distribution.
        
        """
        
        # ensure to overwrite existing
        if FrechetOrGEV == 'Frechet':
            self.ML_estimators = Frechet_ML_estimators(TimeSeries=self.TimeSeries)
            if np.shape(self.PWM_estimators.values) == np.shape([]) and initParams == 'auto':
                self.get_PWM_estimation()
            self.ML_estimators.get_ML_estimation(PWM_estimators=self.PWM_estimators, initParams = initParams, r=r)
            self.ML_estimators.get_statistics(alpha_true=1/self.gamma_true)

        if FrechetOrGEV == 'GEV':
            self.ML_estimators = ML_estimators(TimeSeries=self.TimeSeries)
            if np.shape(self.PWM_estimators.values) == np.shape([]) and initParams == 'auto':
                self.get_PWM_estimation()
            self.ML_estimators.get_ML_estimation(PWM_estimators=self.PWM_estimators, initParams = initParams, r=r)
            self.ML_estimators.get_statistics(gamma_true=self.gamma_true)
        

############################### DATA CLASS ###############################
class Data:
    r"""
    A class for analyzing data with block maxima and high order statistics.

    This class provides methods to extract block maxima and high order statistics from a given dataset. 
    It also supports Maximum Likelihood (ML) estimation for the parameters of either the Frechet or GEV distributions.

    Parameters
    ----------
    values : list or numpy.ndarray
        The dataset from which block maxima and high order statistics are extracted. This should be a 1D array 
        or list of values representing the time series or data.

    Attributes
    ----------
    values : list or numpy.ndarray
        The dataset on which operations are performed.
    len : int
        The length of the dataset.
    blockmaxima : list
        List to store the block maxima extracted from the dataset.
    bm_indices : list
        List of indices corresponding to the positions of the block maxima in the original dataset.
    high_order_stats : list
        List to store the high order statistics extracted from the dataset.
    ML_estimators : ML_estimators_data
        Object that stores and handles the ML estimation results for the Frechet or GEV parameters.

    Methods
    -------
    get_blockmaxima(block_size=2, stride='DBM'):
        Extracts block maxima from the dataset.
    get_HOS(orderstats=2, block_size=2, stride='DBM'):
        Extracts high order statistics from the dataset.
    get_ML_estimation(r=None, FrechetOrGEV='GEV'):
        Computes ML estimations for the Frechet or GEV parameters.

    Example
    -------
    >>> # Initialize the Data object with a dataset
    >>> data = Data([2.5, 3.1, 1.7, 4.6, 5.3, 2.2, 6.0])
    >>> # Extract block maxima using a block size of 2
    >>> data.get_blockmaxima(block_size=2, stride='DBM')
    >>> print(data.blockmaxima)
    >>> # Extract second-highest order statistics (HOS) with the same block size and stride
    >>> data.get_HOS(orderstats=2, block_size=2, stride='DBM')
    >>> print(data.high_order_stats)
    >>> # Perform ML estimation using the extracted HOS, choosing between Frechet and GEV
    >>> data.get_ML_estimation(FrechetOrGEV='GEV')
    >>> print(data.ML_estimators.values)
    """
    
    def __init__(self, values):
        """
        Initialize the Data class with the dataset.

        Parameters
        ----------
        values : list or numpy.ndarray
            The dataset on which to perform analysis.
        """
        self.values = values
        self.len = len(values)
        self.blockmaxima = []
        self.bm_indices = []
        self.high_order_stats = []
    
    def __len__(self):
        return self.len
    
        
    def get_blockmaxima(self, block_size=2, stride='DBM'):
        r"""
        Extract block maxima from the dataset.

        Block maxima are the largest values in each block of the dataset, where the block size 
        and the stride (step size) determine how the blocks are divided.

        Parameters
        ----------
        block_size : int, optional
            The size of blocks for maxima extraction. Default is 2.
        stride : str or int, optional
            The type of stride to use when extracting blocks. Choose from:
            - 'SBM': Sliding Block Maxima (step size 1)
            - 'DBM': Disjoint Block Maxima (non-overlapping blocks)
            - int: Specifies the step size directly.
            Default is 'DBM'.

        Returns
        -------
        None
            The block maxima and their corresponding indices are stored in the `blockmaxima` 
            and `bm_indices` attributes.
        """
        # ensure to overwrite existing
        self.blockmaxima = []
        self.block_size = block_size
        self.stride = stride
        self.bm_indices, self.blockmaxima = extract_BM(self.values, block_size=block_size, stride=stride, return_indices=True)
    

    def get_HOS(self, orderstats = 2, block_size=2, stride='DBM'):
        r"""
        Extract high order statistics (HOS) from the dataset.

        High order statistics refer to statistics of interest beyond the maximum (e.g., second-highest, 
        third-highest). This method extracts these statistics from blocks of the dataset.

        Parameters
        ----------
        orderstats : int, optional
            The order of the statistic to extract (e.g., 2 for second-highest). Default is 2.
        block_size : int, optional
            The size of blocks from which to extract the statistics. Default is 2.
        stride : str or int, optional
            The type of stride to use when extracting blocks. Choose from:
            - 'SBM': Sliding Block Maxima (step size 1)
            - 'DBM': Disjoint Block Maxima (non-overlapping blocks)
            - int: Specifies the step size directly.
            Default is 'DBM'.

        Returns
        -------
        None
            The high order statistics are stored in the `high_order_stats` attribute.
        """
        # ensure to overwrite existing
        self.high_order_stats = []
        self.block_size = block_size
        self.stride = stride
        self.high_order_stats = extract_HOS(self.values, orderstats=orderstats, block_size=block_size, stride=stride)

    def get_ML_estimation(self, r=None, FrechetOrGEV = 'GEV'):
        r"""
        Compute Maximum Likelihood (ML) estimations for the Frechet or GEV parameters.

        This method computes ML estimators for the parameters of either the Frechet or GEV distribution based 
        on the high order statistics extracted from the data. If no high order statistics are available, it 
        will first extract them.

        Parameters
        ----------
        r : int, optional
            The number of order statistics to use in the ML estimation. If not specified, it uses all the extracted statistics.
        FrechetOrGEV : str, optional
            The type of distribution to use for the ML estimation. Choose between 'Frechet' and 'GEV'. Default is 'GEV'.

        Returns
        -------
        None
            The ML estimators are stored in the `ML_estimators` attribute.
        """
        if np.array(self.high_order_stats).shape == (0,):
            # potentially use parameters optained by getting blockmaxima
            self.get_HOS(orderstats=1, block_size=self.block_size, stride=self.stride)
        self.ML_estimators = ML_estimators_data(self.high_order_stats)
        self.ML_estimators.get_ML_estimation(FrechetOrGEV=FrechetOrGEV, r=r)


class ML_estimators_data:
    r"""
    Class for performing Maximum Likelihood (ML) estimation on high order statistics.

    This class calculates the Maximum Likelihood Estimators (MLE) for either the Generalized Extreme Value (GEV) 
    distribution or the Frechet distribution based on the provided high order statistics. It also contains methods 
    to compute relevant statistics from the estimators.

    Parameters
    ----------
    high_order_stats : numpy.ndarray
        The array of high order statistics on which ML estimation will be performed.

    Attributes
    ----------
    high_order_stats : numpy.ndarray
        The high order statistics provided during initialization.
    values : numpy.ndarray
        The ML estimators (parameters) calculated for the GEV or Frechet distribution. If `FrechetOrGEV='GEV'`, 
        the array contains [gamma, mu, sigma]. If `FrechetOrGEV='Frechet'`, the array contains [alpha, sigma].
    statistics : dict
        A dictionary to store computed statistics related to the ML estimators.

    Methods
    -------
    get_ML_estimation(FrechetOrGEV='GEV', r=None):
        Computes the ML estimators for either the GEV or Frechet distribution.
    get_statistics(gamma_true):
        (Placeholder) Computes additional statistics such as bootstrapped confidence intervals for the estimators.

    Example
    -------
    >>> # Assuming you have high order statistics stored in `hos`
    >>> hos = np.array([[0.5, 1.0], [1.5, 2.0], [1.2, 2.2], [2.0, 3.0]])
    >>> # Create an instance of the ML_estimators_data class
    >>> ml = ML_estimators_data(hos)
    >>> # Perform ML estimation for the GEV distribution
    >>> ml.get_ML_estimation(FrechetOrGEV='GEV', r=2)
    >>> # Print the estimated parameters (gamma, mu, sigma)
    >>> print(ml.values)
    >>> # Perform ML estimation for the Frechet distribution
    >>> ml.get_ML_estimation(FrechetOrGEV='Frechet', r=2)
    >>> # Print the estimated parameters (alpha, sigma)
    >>> print(ml.values)
    """

    def __init__(self, high_order_stats):
        r"""
        Initialize the ML_estimators_data class with high order statistics.

        Parameters
        ----------
        high_order_stats : numpy.ndarray
            The high order statistics on which ML estimation will be performed.
        """

        self.high_order_stats = high_order_stats
        self.values = []
        self.statistics = {}
    
    def __len__(self):
        return len(self.values)

    def get_ML_estimation(self, FrechetOrGEV='GEV', r=None):
        r"""
        Perform Maximum Likelihood (ML) estimation for the GEV or Frechet distribution.

        This method computes ML estimators for either the Generalized Extreme Value (GEV) or Frechet distribution 
        based on the provided high order statistics. It minimizes the negative log-likelihood to estimate the 
        distribution parameters.

        Parameters
        ----------
        FrechetOrGEV : str, optional
            The type of distribution for which the ML estimators are calculated. Options are 'GEV' or 'Frechet'. 
            Default is 'GEV'.
        r : int, optional
            The number of order statistics to use for the ML estimation. If not specified, all available order 
            statistics are used.

        Returns
        -------
        None
            The estimated parameters are stored in the `values` attribute as a NumPy array. For the GEV distribution, 
            this will be [gamma, mu, sigma]. For the Frechet distribution, this will be [alpha, sigma].

        Raises
        ------
        ValueError
            If an invalid value is provided for the `FrechetOrGEV` parameter.
        """
        if FrechetOrGEV == 'GEV':
            def cost(params):
                gamma, mu, sigma = params
                cst = - log_likelihood(self.high_order_stats, gamma=gamma, mu=mu, sigma=sigma, r=r)
                return cst
            # for now: hard-coded init params [1,1,1], as it should not matter too much
            #results = misc.minimize(cost, [1,1,1], method='Nelder-Mead', bounds=((0,np.inf),(-np.inf,np.inf),(0,np.inf)))
            # results = misc.minimize(cost, [1,1,1], method='COBYLA', bounds=((0,np.inf),(-np.inf,np.inf),(0,np.inf)))
            results = misc.minimize(cost, [1,1,1], method='L-BFGS-B', bounds=[(None, None), (None, None), (1e-5, None)])
        
            gamma, mu, sigma = results.x
            self.values = np.array([gamma, mu, sigma])

        elif FrechetOrGEV == 'Frechet':
            # alternative: use Lemma 3.1 for MLE computation if r=2
            if r==2: 
                sec, mxm = self.high_order_stats.T[0], self.high_order_stats.T[1]
                k = len(mxm)
                mask = sec > 0
                def Psi(a):
                    return 2/a + 2 * np.mean(sec[mask]**(-a))**(-1)*np.mean(sec[mask]**(-a)*np.log(sec[mask])) - np.mean(np.log(sec[mask]*mxm[mask]))
                from scipy.optimize import root_scalar
                alpha = root_scalar(Psi, bracket=[1e-5, 100]).root
                sigma = 2**(1/alpha)*np.mean(sec[mask]**(-alpha))**(-1/alpha)
            else:
                def cost(params):
                    alpha, sigma = params
                    cst = - Frechet_log_likelihood(self.high_order_stats, alpha=alpha, sigma=sigma, r=r)
                    return cst
                # for now: hard-coded init params [1,1], as it should not matter too much
                # results = misc.minimize(cost, [1,1], method='COBYLA', bounds=((0,np.inf),(0,np.inf)))
                results = misc.minimize(cost, [1,1], method='Nelder-Mead', bounds=((1e-5,np.inf)  ,(1e-5,np.inf)))
                alpha, sigma = results.x
            
            self.values = np.array([alpha, sigma])
        else:
            raise ValueError("FrechetOrGEV has to be 'Frechet' or 'GEV', but is ", FrechetOrGEV)

    def bootstrap(self, n_boot=500, set_seed=True, FrechetOrGEV = 'GEV', r=None):
        bst_samp = []
        for _ in range(n_boot):
            if set_seed:
                np.random.seed(_)
            l = len(self.high_order_stats)
            idx = np.random.choice(np.arange(l), size=l,replace=True)
            new_data = np.array([self.high_order_stats[i] for i in idx])
            if FrechetOrGEV == 'GEV':
                def cost(params):
                    gamma, mu, sigma = params
                    cst = - log_likelihood(self.high_order_stats, gamma=gamma, mu=mu, sigma=sigma, r=r)
                    return cst
                # for now: hard-coded init params [1,1,1], as it should not matter too much
                results = misc.minimize(cost, [1,1,1], method='Nelder-Mead', bounds=((0,np.inf),(-np.inf,np.inf),(0,np.inf)))
                # results = misc.minimize(cost, [1,1,1], method='COBYLA', bounds=((0,np.inf),(-np.inf,np.inf),(0,np.inf)))
                gamma, mu, sigma = results.x
                bst_samp.append(np.array([gamma, mu, sigma]))

            elif FrechetOrGEV == 'Frechet':
                def cost(params):
                    alpha, sigma = params
                    cst = - Frechet_log_likelihood(self.high_order_stats, alpha=alpha, sigma=sigma, r=r)
                    return cst
                # for now: hard-coded init params [1,1], as it should not matter too much
                # results = misc.minimize(cost, [1,1], method='COBYLA', bounds=((0,np.inf),(0,np.inf)))
                results = misc.minimize(cost, [1,1], method='Nelder-Mead', bounds=((0,np.inf),(0,np.inf)))
        
                alpha, sigma = results.x
                
                bst_samp.append(np.array([alpha, sigma]))
            else:
                raise ValueError("FrechetOrGEV has to be 'Frechet' or 'GEV', but is ", FrechetOrGEV)
        
        self.bootstrap_sample = np.array(bst_samp)


