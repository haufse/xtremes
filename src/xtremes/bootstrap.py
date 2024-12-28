import numpy as np
from collections import defaultdict
import xtremes.HigherOrderStatistics as hos
import xtremes.miscellaneous as misc
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# plt.rcParams.update({
#     # "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "font.size": 16,
#     "figure.figsize": (10, 6),
# })

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
    - In the 'SBM' setting, the circmax() method introduced by Bücher and Staud 2024 is used.

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
        resh_s = sample[:(k//2)*2*bs].reshape((k//2,2*bs))
        resh_s_a = np.append(resh_s, resh_s[:,:bs-1],axis=1)
        f = lambda x: hos.extract_BM(x, bs, 'SBM')
        return np.apply_along_axis(f, 1, resh_s_a)
    else:
        raise ValueError('No valid stride specified.')

def uniquening(circmaxs, stride = 'DBM'):
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
    if stride == 'DBM':
        return circmaxs
    if stride == 'SBM':
        return [np.unique(x,return_counts=True) for x in np.unique(circmaxs)]

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
    boot_samp = [xx[i] for i in inds] 
    return boot_samp

def aggregate_boot(boot_samp, stride='DBM'):
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
    if stride == 'DBM':
        return np.array(np.unique(boot_samp, return_counts=True))
    
    if stride == 'SBM':
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


# auxiliary function to parallelize the bootstrap
def bootstrap_worker(args):
    r"""
    Auxiliary function to perform a single bootstrap resampling and MLE estimation.

    This function is designed to be used in parallelized bootstrap procedures. It takes arguments for a single 
    bootstrap iteration, performs resampling on the given block maxima, estimates MLE parameters using the 
    specified distribution type, and returns the results.

    Parameters
    ----------
    args : tuple
        A tuple containing the following elements:
        - idx (int): The iteration index, used for setting the random seed if `set_seeds` is True.
        - set_seeds (bool): Whether to set the random seed for reproducibility.
        - circmaxs (list or numpy.ndarray): The block maxima dataset to be resampled.
        - aggregate_boot (callable): A function to aggregate the resampled data.
        - ML_estimators_data (callable): A function or class to compute MLE parameters on the aggregated data.
        - dist_type (str): The distribution type for MLE estimation ('Frechet' or 'GEV').

    Returns
    -------
    numpy.ndarray
        The MLE parameter estimates for the current bootstrap sample.

    Notes
    -----
    - This function is designed to be compatible with `ProcessPoolExecutor` or other parallel processing tools.
    - The random seed is set per iteration to ensure reproducibility when `set_seeds` is True.

    Example
    -------
    >>> args = (0, True, circmaxs, aggregate_boot, ML_estimators_data, 'GEV')
    >>> bootstrap_worker(args)
    array([param1, param2, param3])  # Example output for GEV distribution
    """

    idx, set_seeds, circmaxs, aggregate_boot, ML_estimators_data, dist_type = args
    if set_seeds:
        np.random.seed(idx)
    # Bootstrap sample
    boot = Bootstrap(circmaxs)
    aggregated_data = aggregate_boot(boot)
    # Create an ML_Estimator for the current bootstrap sample
    d = np.array([np.repeat(aggregated_data[0], aggregated_data[1].astype(int))]).T
    estimator = ML_estimators_data(d)
    # estimator = ML_estimators_data((aggregated_data[0], aggregated_data[1].astype(int)))  
    estimator.get_ML_estimation(FrechetOrGEV=dist_type)
    return estimator.values




class FullBootstrap:
    r"""
    A class to perform bootstrapping of Maximum Likelihood Estimates (MLE) for Fréchet or GEV distributions.

    This class performs block maxima extraction from an initial sample using either disjoint or sliding blocks.
    It applies a bootstrap resampling procedure to estimate the variability of the MLE parameters for the specified
    distribution type (Fréchet or GEV). The bootstrap method is parallelized for efficiency and supports reproducibility
    through optional seed setting.

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
    data : hos.Data
        The `hos.Data` object containing the original dataset and its MLE results.
    MLEvals : numpy.ndarray
        The MLE estimates from the original dataset before bootstrapping.
    values : numpy.ndarray
        MLE estimates for each bootstrap sample after running the `run_bootstrap` method.
    statistics : dict
        Dictionary containing summary statistics (mean and standard deviation) of the bootstrap estimates.

    Methods
    -------
    run_bootstrap(num_bootstraps=100, set_seeds=False, max_workers=1)
        Runs the bootstrap procedure in parallel and calculates the MLE estimates for each bootstrap sample.

    Example
    -------
    >>> sample = np.random.rand(100)
    >>> bootstrap = FullBootstrap(sample, bs=10, stride='DBM', dist_type='Frechet')
    >>> bootstrap.run_bootstrap(num_bootstraps=100, set_seeds=True, max_workers=4)
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
        # just for comparison and CIs:
        self.data = hos.Data(initial_sample)
        self.data.get_HOS(orderstats = 1, block_size=bs, stride=stride)
        self.data.get_ML_estimation(FrechetOrGEV='GEV')
        self.MLEvals = self.data.ML_estimators.values

    def run_bootstrap(self, num_bootstraps=100, set_seeds=False, max_workers=1):
        r"""
        Run the bootstrap resampling procedure in parallel.

        This method resamples the block maxima dataset, estimates the MLE parameters for each bootstrap sample,
        and computes summary statistics (mean and standard deviation) of the bootstrap estimates. The computation
        is parallelized using `ProcessPoolExecutor` with an adjustable number of worker processes.

        Parameters
        ----------
        num_bootstraps : int, optional
            Number of bootstrap samples to generate. Default is 100.
        set_seeds : bool, optional
            If True, sets the random seed for reproducibility in each bootstrap iteration. Default is False.
        max_workers : int, optional
            Maximum number of worker processes to use for parallelization. Default is 1 (no parallelism).
            Set to `None` to use all available CPU cores.

        Returns
        -------
        None
            Results are stored in the `values` attribute and summary statistics in the `statistics` attribute.

        Example
        -------
        >>> bootstrap.run_bootstrap(num_bootstraps=500, set_seeds=True, max_workers=4)
        >>> bootstrap.statistics['mean']  # Access the mean of bootstrap estimates
        >>> bootstrap.statistics['std']   # Access the standard deviation of bootstrap estimates
        """

        # Prepare arguments for the worker function
        args = [
            (idx, set_seeds, self.circmaxs, aggregate_boot, hos.ML_estimators_data, self.dist_type)
            for idx in range(num_bootstraps)
        ]
        
        estimates = []
        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm for progress bar
            results = list(tqdm(executor.map(bootstrap_worker, args), total=num_bootstraps))
        
        # Collect results
        estimates.extend(results)
        self.values = np.array(estimates)
        self.statistics = {
            'mean': np.mean(self.values,axis=0),
            'std': np.std(self.values,axis=0),
        }

   
    
    def get_CI(self, alpha=0.05, method='bootstrap'):
        r"""
        Compute the confidence interval (CI) for the Maximum Likelihood Estimate (MLE) parameters 
        based on bootstrap samples.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence interval. Default is 0.05, 
            corresponding to a 95% confidence interval.
            
        method : str, optional
            Method to compute the confidence interval. Two options are available:
            - 'symmetric': The confidence interval is computed using the symmetric quantiles.
            - 'minimal_width': The confidence interval is computed by finding the minimal-width 
            interval that contains (1 - alpha) proportion of the bootstrap distribution.
            The default is 'symmetric'.

        Returns
        -------
        numpy.ndarray
            A 2D array with shape (n_parameters, 2) containing the lower and upper bounds of 
            the confidence interval for each parameter. The first column represents the lower 
            bounds, and the second column represents the upper bounds.

        Notes
        -----
        The confidence intervals are based on bootstrap estimates of the MLE parameters, which 
        means the confidence intervals are derived from the empirical distribution of the parameter 
        estimates obtained from multiple bootstrap samples.

        There are two methods available for calculating the confidence intervals:
        - 'symmetric': This method takes the alpha/2 and (1 - alpha/2) quantiles of the bootstrap 
        distribution for each parameter. It is based on the assumption that the distribution 
        is approximately symmetric and works well when the bootstrap distribution is roughly normal.
        - 'minimal_width': This method identifies the interval with the minimal width that contains 
        (1 - alpha) proportion of the bootstrap samples. It is particularly useful when the 
        bootstrap distribution is skewed or not symmetric.
        """
        if method == 'bootstrap':
            # the standard CI reported when performing a bootstrap
            l = np.quantile(self.values, alpha/2, axis=0)
            u = np.quantile(self.values, (1-alpha/2), axis=0)
            lower = 2 * self.MLEvals - u
            upper = 2 * self.MLEvals - l


        if method == 'symmetric':
            lower = np.quantile(self.values, alpha/2, axis=0)
            upper = np.quantile(self.values, (1-alpha/2), axis=0)
        if method == 'minimal_width':
            sorted_values = np.sort(self.values, axis=0)
            n = len(sorted_values)
            best_interval = np.zeros((2, sorted_values.shape[1]))           
            for param_idx in range(sorted_values.shape[1]):
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
                
                best_interval[0][param_idx] = param_best_interval[0]
                best_interval[1][param_idx] = param_best_interval[1]
            
            lower, upper = best_interval
        self.CI = np.stack([lower, upper], axis=1)
    
    def plot_bootstrap(self, param_idx=0, param_name='gamma', bins=30, output_file=None, show=True):
        r"""
        Plot the bootstrap distribution for a specified parameter.

        Parameters
        ----------
        param_idx : int, optional
            Index of the parameter to plot (0 for the first parameter, 1 for the second, etc.). Default is 0.
        bins : int, optional
            Number of bins to use for the histogram. Default is 30.

        Notes
        -----
        This method generates a histogram of the bootstrap estimates for the specified parameter and overlays
        the mean and confidence interval.
        """

        if not hasattr(self, 'values'):
            raise RuntimeError("You must run the bootstrap procedure before plotting.")

        param_values = self.values[:,param_idx]
        mean_value = self.statistics['mean'][param_idx]
        ci_lower, ci_upper = self.CI[param_idx, :]

        plt.hist(param_values, bins=bins, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label='Mean')
        plt.axvline(ci_lower, color='green', linestyle='dashed', linewidth=2, label='95% CI Lower')
        plt.axvline(ci_upper, color='green', linestyle='dashed', linewidth=2, label='95% CI Upper')
        plt.title(f'Bootstrap Distribution for Parameter {param_name}')
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        plt.legend()
        if output_file:
            plt.savefig(output_file)
            plt.close()
        if show:
            plt.show()
