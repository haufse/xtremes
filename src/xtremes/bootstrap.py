import numpy as np
from collections import defaultdict
import xtremes.HigherOrderStatistics as hos
from scipy.optimize import minimize

def circmax(sample, bs=10, stride='DBM'):
    r"""
    Extract the block maxima (BM) from a given sample using different stride methods.

    Parameters
    ----------
    sample : numpy.ndarray
        A 1D array containing the sample from which block maxima will be extracted.
    bs : int, optional
        The block size (number of observations per block) used to divide the sample for block maxima extraction. Default is 10.
    stride : {'DBM', 'SBM'}, optional
        The stride method used for extracting block maxima:
        - 'DBM' (Disjoint Block Maxima): Extracts maxima from non-overlapping blocks.
        - 'SBM' (Sliding Block Maxima): Extracts maxima using overlapping blocks. Default is 'DBM'.

    Returns
    -------
    numpy.ndarray
        A 1D or 2D array containing the block maxima extracted from the sample. The result depends on the stride method used:
        - For 'DBM', returns a 1D array of block maxima.
        - For 'SBM', returns a 2D array where each row contains the block maxima extracted from overlapping blocks.

    Raises
    ------
    ValueError
        If an invalid stride method is specified.

    Notes
    -----
    - 'DBM' (Disjoint Block Maxima) extracts block maxima from non-overlapping blocks of size `bs`.
    - 'SBM' (Sliding Block Maxima) creates overlapping blocks, effectively increasing the number of block maxima compared to 'DBM'.
    - In th 'SBM' setting, the circmax() method introduced by Bücher and Staud 2024 is used.

    References
    ----------
    Bücher, A., & Staud, T. (2024). Bootstrapping Estimators based on the Block Maxima Method. 
    arXiv preprint arXiv:2409.05529.
    
    Example
    -------
    >>> sample = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> circmax(sample, bs=5, stride='DBM')
    array([5, 10])
    
    >>> circmax(sample, bs=3, stride='SBM')
    array([[3, 6, 9],
           [4, 7, 10]])
"""

    if stride == 'DBM':
        return hos.extract_BM(sample, bs, stride, return_indices=False)
    if stride == 'SBM':
        k = len(sample) // bs
        resh_s = sample[:k*bs].reshape((k//2,2*bs))
        resh_s_a = np.append(resh_s, resh_s[:,:bs-1],axis=1)
        f = lambda x: hos.extract_BM(x, bs, 'SBM')
        return np.apply_along_axis(f, 1, resh_s_a)
    else:
        raise ValueError('No valid stride specified.')

def uniquening(circmaxs):
    r"""
    Identify unique values and their counts from a list of arrays.

    Parameters
    ----------
    circmaxs : numpy.ndarray
        A NumPy array containing block maxima values extracted from a sample.

    Returns
    -------
    list of tuples
        A list where each element is a tuple containing two NumPy arrays:
        - The first array contains the unique values from the corresponding row in `circmaxs`.
        - The second array contains the counts of each unique value.

    """

    return [np.unique(x,return_counts=True) for x in circmaxs]

def Bootstrap(xx):
    r"""
    Generate a bootstrap sample by resampling with replacement from the input data.

    Parameters
    ----------
    xx : list or numpy.ndarray
        The input sample to resample from.

    Returns
    -------
    list
        A new sample of the same size, created by randomly selecting elements from `xx` with replacement.

    Notes
    -----
    This function creates a bootstrap sample, which is commonly used in statistical resampling methods to estimate 
    the variability of a statistic.


    Example
    -------
    >>> sample = [1, 2, 3, 4, 5]
    >>> Bootstrap(sample)
    [2, 5, 3, 1, 2]  # Example output, actual result may vary
    """
    l = len(xx)
    
    # Sample indices with replacement
    inds = np.random.choice(np.arange(l), size=l, replace=True)
    
    # Create the bootstrapped sample list
    boot_samp = [xx[i] for i in inds]  # Adjust indexing since Python is 0-based
    
    return boot_samp

def aggregate_boot(boot_samp):
    r"""
    Aggregate counts of unique values from a list of tuples containing values and their counts.

    Parameters
    ----------
    boot_samp : list of tuples
        Each tuple contains two arrays: the first with values and the second with corresponding counts.

    Returns
    -------
    numpy.ndarray
        A 2D array with two columns: the first column contains unique values, and the second column contains the aggregated counts.

    Example
    -------
    >>> boot_samp = [(np.array([1, 2, 3]), np.array([1, 1, 2])), (np.array([2, 3]), np.array([2, 1]))]
    >>> aggregate_boot(boot_samp)
    array([[1, 1],
           [2, 3],
           [3, 3]])
"""
    
    # Dictionary to store the counts
    value_counts = defaultdict(int)

    # Summing the occurrences
    for values, counts in boot_samp:
        for value, count in zip(values, counts):
            value_counts[value] += count

    # Convert the result into a sorted 2D NumPy array
    sorted_values = sorted(value_counts.items())
    result_array = np.array(sorted_values)
    
    return result_array

class ML_Estimator:
    r"""
    A class to perform Maximum Likelihood Estimation (MLE) for the Fréchet and Generalized Extreme Value (GEV) distributions on Bootstrap sample.

    This class takes aggregated data where each value has a corresponding count and provides methods to compute
    the MLE of parameters for the 2-parameter Fréchet distribution and the 3-parameter GEV distribution.

    Parameters
    ----------
    aggregated_data : numpy.ndarray
        A 2D array where the first column contains unique values and the second column contains their respective counts.

    Attributes
    ----------
    data : numpy.ndarray
        The unique values from the input aggregated data.
    counts : numpy.ndarray
        The corresponding counts for the values from the input aggregated data.

    Methods
    -------
    maximize_frechet(initial_params=[1, 1])
        Maximizes the log-likelihood for the Fréchet distribution and returns optimized parameters.
    maximize_gev(initial_params=[1, 1, 1])
        Maximizes the log-likelihood for the GEV distribution and returns optimized parameters.

    Example
    -------
    >>> aggregated_data = np.array([[5, 2], [10, 3], [15, 4]])
    >>> estimator = ML_Estimator(aggregated_data)
    >>> frechet_params = estimator.maximize_frechet()
    >>> gev_
    """

    def __init__(self, aggregated_data):
        r"""
        Initialize the ML_Estimator with aggregated data.

        Parameters
        ----------
        aggregated_data : numpy.ndarray
            A 2D array where the first column contains the unique values and the second column contains the counts.
        """
        self.data = aggregated_data[:, 0]  # Values
        self.counts = aggregated_data[:, 1]  # Counts
    
    def _frechet_log_likelihood(self, params):
        r"""
        Compute the log-likelihood for the 2-parametric Fréchet distribution.

        Parameters
        ----------
        params : list or tuple
            A list containing [alpha, sigma], the shape and scale parameters of the Fréchet distribution.

        Returns
        -------
        float
            The negative log-likelihood value for the given parameters.

        Notes
        -----
        This method computes the log-likelihood for the Fréchet distribution and is used for internal optimization.
        """
        alpha, sigma = params
        if alpha <= 0 or sigma <= 0:
            return np.inf  # Invalid parameter values
        
        values = self.data
        counts = self.counts
        
        # Fréchet log-likelihood
        log_likelihood = (
            np.dot(counts, (np.log(alpha/sigma) - (alpha + 1) * np.log(values / sigma) - (values / sigma) ** (-alpha)))
        )
        
        # We negate the log-likelihood for minimization
        return -log_likelihood
    
    def maximize_frechet(self, initial_params= [1,1]):
        r"""
        Maximize the log-likelihood for the Fréchet distribution.

        Parameters
        ----------
        initial_params : list, optional
            Initial guess for the [alpha, sigma] parameters. Default is [1, 1].

        Returns
        -------
        numpy.ndarray
            The optimized [alpha, sigma] parameters.

        Notes
        -----
        This method uses the `scipy.optimize.minimize` function with the L-BFGS-B algorithm to maximize the 
        log-likelihood of the Fréchet distribution.
        """
        result = minimize(self._frechet_log_likelihood, initial_params, method='L-BFGS-B', bounds=[(1e-5, None), (1e-5, None)])
        return result.x  # Optimized parameters
    
    def _gev_log_likelihood(self, params):
        r"""
        Compute the log-likelihood for the 3-parametric Generalized Extreme Value (GEV) distribution.

        Parameters
        ----------
        params : list or tuple
            A list containing [mu, sigma, xi], the location, scale, and shape parameters of the GEV distribution.

        Returns
        -------
        float
            The negative log-likelihood value for the given parameters.

        Notes
        -----
        - If the shape parameter xi is close to zero, the distribution is treated as the Gumbel distribution.
        - This method computes the log-likelihood for the GEV distribution and is used for internal optimization.
        """
        mu, sigma, xi = params
        if sigma <= 0:
            return np.inf  # Invalid parameter values

        values = self.data
        counts = self.counts

        # Calculate the scaled values
        z = (values - mu) / sigma

        # Check if xi is close to zero (Gumbel case)
        if np.abs(xi) < 1e-3:
            # Gumbel distribution log-likelihood
            t = np.exp(-z)
            log_likelihood = (
                np.dot(counts, -(np.log(sigma) + z + t))
            )
        else:

            t = 1 + xi * z

            if np.any(t <= 0):
                return np.inf  # Invalid values lead to complex numbers

            # GEV log-likelihood
            log_likelihood = (
                np.dot(counts, -(np.log(sigma) + ((1 + 1/xi) * np.log(t)) + t ** (-1/xi)))
            )

        # We negate the log-likelihood for minimization
        return -log_likelihood
    
    def maximize_gev(self, initial_params=[1,1,1]):
        r"""
        Maximize the log-likelihood for the Generalized Extreme Value (GEV) distribution.

        Parameters
        ----------
        initial_params : list, optional
            Initial guess for the [mu, sigma, xi] parameters. Default is [1, 1, 1].

        Returns
        -------
        numpy.ndarray
            The optimized [mu, sigma, xi] parameters.

        Notes
        -----
        This method uses the `scipy.optimize.minimize` function with the L-BFGS-B algorithm to maximize the 
        log-likelihood of the GEV distribution.
        """
        result = minimize(self._gev_log_likelihood, initial_params, method='L-BFGS-B', bounds=[(None, None), (1e-5, None), (None, None)])
        return result.x  # Optimized parameters


class FullBootstrap:
    r"""
    A class to perform bootstrapping of Maximum Likelihood Estimates (MLE) for Fréchet or GEV distributions.

    This class takes an initial sample and applies a bootstrap resampling procedure to estimate the variability of
    the MLE parameters for the specified distribution type (Fréchet or GEV). It uses block maxima extraction methods
    with disjoint or sliding blocks.

    Parameters
    ----------
    initial_sample : list or numpy.ndarray
        The initial dataset from which block maxima will be extracted and bootstrapped.
    bs : int, optional
        Block size for the block maxima extraction. Default is 10.
    stride : {'DBM', 'SBM'}, optional
        Stride type for block maxima extraction:
        - 'DBM' (Disjoint Block Maxima): Non-overlapping blocks.
        - 'SBM' (Sliding Block Maxima): Overlapping blocks.
        Default is 'DBM'.
    dist_type : {'Frechet', 'GEV'}, optional
        Distribution type to estimate the parameters for:
        - 'Frechet': Estimate parameters for the 2-parametric Fréchet distribution.
        - 'GEV': Estimate parameters for the 3-parametric Generalized Extreme Value (GEV) distribution.
        Default is 'Frechet'.

    Attributes
    ----------
    circmaxs : list
        The block maxima extracted from the initial sample using the specified block size and stride.
    values : list
        MLE estimates for each bootstrap sample after running the `run_bootstrap` method.
    statistics : dict
        Dictionary containing summary statistics (mean and standard deviation) of the bootstrap estimates.

    Methods
    -------
    run_bootstrap(num_bootstraps=100)
        Runs the bootstrap procedure and returns the MLE estimates for each bootstrap sample.

    Example
    -------
    >>> sample = np.random.rand(100)
    >>> bootstrap = FullBootstrap(sample, bs=10, stride='DBM', dist_type='Frechet')
    >>> bootstrap.run_bootstrap(num_bootstraps=100)
    >>> bootstrap.statistics['mean']  # Mean of bootstrap estimates
    >>> bootstrap.statistics['std']   # Standard deviation of bootstrap estimates
    """

    def __init__(self, initial_sample, bs=10, stride='DBM', dist_type='Frechet'):
        r"""
        Initialize the FullBootstrap instance.

        Parameters
        ----------
        initial_sample : list or numpy.ndarray
            The initial dataset to be bootstrapped.
        bs : int, optional
            Block size for the circmax function. Default is 10.
        stride : {'DBM', 'SBM'}, optional
            Stride type for the circmax function (either 'DBM' or 'SBM'). Default is 'DBM'.
        dist_type : {'Frechet', 'GEV'}, optional
            Distribution type ('Frechet' or 'GEV') to calculate the MLE. Default is 'Frechet'.
        """
        self.initial_sample = initial_sample
        self.bs = bs
        self.stride = stride
        self.dist_type = dist_type
        # Extract block maxima using the predefined `circmax` function
        self.circmaxs = uniquening(circmax(self.initial_sample, bs=self.bs, stride=self.stride))

    def run_bootstrap(self, num_bootstraps=100):
        r"""
        Run the bootstrap procedure and return the MLE estimates for each bootstrap sample.

        Parameters
        ----------
        num_bootstraps : int, optional
            Number of bootstrap iterations to run. Default is 100.

        Returns
        -------
        list
            MLE estimates for each bootstrap sample.

        Notes
        -----
        The procedure uses bootstrapping to estimate the variability of the MLE parameters based on resampled
        block maxima. The MLE parameters are estimated for either the Fréchet or GEV distribution depending
        on the specified distribution type during initialization.

        After running this method, the `values` and `statistics` attributes will contain the results of the bootstrap.

        Example
        -------
        >>> sample = np.random.rand(100)
        >>> bootstrap = FullBootstrap(sample, bs=10, stride='DBM', dist_type='Frechet')
        >>> bootstrap.run_bootstrap(num_bootstraps=100)
        >>> bootstrap.statistics['mean']  # Mean of bootstrap estimates
        >>> bootstrap.statistics['std']   # Standard deviation of bootstrap estimates
        """
        estimates = []
        
        for _ in range(num_bootstraps):

            # Bootstrap sample
            boot = Bootstrap(self.circmaxs)
            aggregated_data = aggregate_boot(boot)
            # Create an ML_Estimator for the current bootstrap sample
            estimator = ML_Estimator(aggregated_data)
            
            # Compute MLE based on distribution type (Frechet or GEV)
            if self.dist_type == 'Frechet':
                estimate = estimator.maximize_frechet()
            elif self.dist_type == 'GEV':
                estimate = estimator.maximize_gev()
            else:
                raise ValueError('Invalid distribution type specified.')
            
            estimates.append(estimate)
        
        self.values = estimates
        self.statistics = {
            'mean': np.mean(estimates,axis=0),
            'std': np.std(estimates,axis=0),
        }
