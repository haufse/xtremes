import numpy as np
import xtremes.topt as topt

def I_sb(j, bs):
    r"""
    Generate a range of integers representing indices for a sliding block method.

    This function is often used in time series analysis or statistical modeling 
    contexts where data is divided into overlapping or non-overlapping blocks.

    Parameters:
    -----------
    j : int
        The starting index for the block. This typically represents the position 
        in the time series or dataset where the block begins.
        
    bs : int
        The block size, which determines the number of elements in the block 
        and the range of indices generated.

    Returns:
    --------
    numpy.ndarray
        An array of consecutive integers starting from `j` and ending at `j + bs - 1`.
        This array can be used to reference indices of elements within a specific block.

    Example:
    --------
    >>> import numpy as np
    >>> I_sb(3, 5)
    array([3, 4, 5, 6, 7])

    References:
    -----------
    Bücher, A., & Jennessen, T. (2022). Statistical analysis for stationary time series 
    at extreme levels: New estimators for the limiting cluster size distribution. 
    Stochastic Processes and their Applications, 149, 75-106.
    """
    return np.arange(j,j+bs)

def D_n(n, bs):
    r"""
    Generate all pairs of indices representing disjoint blocks in a time series.

    This function identifies pairs of block indices `(i, j)` such that the blocks 
    starting at indices `i` and `j`, each of size `bs`, do not overlap. 
    It is useful in time series analysis for constructing disjoint block structures, 
    particularly in statistical estimators for cluster sizes.

    Parameters:
    -----------
    n : int
        The length of the time series or dataset. This determines the range of 
        valid indices for block positions.
        
    bs : int
        The block size, which specifies the number of elements in each block.
        Blocks are constructed as contiguous subsets of the time series.

    Returns:
    --------
    list of tuples
        A list of tuples `(i, j)`, where `i` and `j` are the starting indices 
        of two disjoint blocks. The blocks `I_sb(i, bs)` and `I_sb(j, bs)` have 
        no overlap, i.e., their intersection is empty.

    Example:
    --------
    >>> import numpy as np
    >>> def I_sb(j, bs):
    >>>     return np.arange(j, j + bs)
    >>> D_n(5, 2)
    [(1, 3), (1, 4), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]

    Notes:
    ------
    - The function uses the helper function `I_sb` to define the range of indices 
      covered by a block starting at a given index.
    - It checks for disjointness using `np.intersect1d`, which computes the 
      intersection of two arrays.
    - This function is critical in methods involving disjoint blocks, such as 
      certain statistical estimators that assume independence between blocks.
    """
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
    r"""
    Calculate the number of exceedances of a given maximum within a specified block.

    This function computes the number of values in the `j`-th block of the data that 
    exceed the `i`-th maximum. The block can be defined using either a disjoint block 
    method (DBM) or a sliding block method (SBM).

    Parameters:
    -----------
    data : array-like
        The time series or dataset from which exceedances are calculated.
        
    maxima : array-like
        An array of maxima values, where `maxima[i-1]` is the reference maximum 
        for which exceedances are counted.
        
    bs : int
        The block size, specifying the number of consecutive elements in each block.

    i : int
        The index (1-based) of the maximum in `maxima` used as the reference for exceedances.

    j : int
        The index (1-based) of the block in the dataset to examine for exceedances.

    stride : {'DBM', 'SBM'}, optional
        Specifies the block method used:
        - 'DBM' (Disjoint Block Method): The blocks are non-overlapping and 
          start at `(j-1)*bs` and end at `j*bs`.
        - 'SBM' (Sliding Block Method): The blocks can overlap and are defined 
          to start at `j` and end at `j+bs`.
        Default is 'DBM'.

    Returns:
    --------
    int
        The number of elements in the specified block that exceed the given maximum.

    Example:
    --------
    >>> import numpy as np
    >>> data = np.array([1, 3, 5, 2, 6, 4, 7, 9])
    >>> maxima = np.array([5, 7])
    >>> exceedances(data, maxima, bs=2, i=1, j=3, stride='DBM')
    1  # One value in the 3rd disjoint block exceeds maxima[0] = 5
    >>> exceedances(data, maxima, bs=3, i=2, j=2, stride='SBM')
    2  # Two values in the sliding block exceed maxima[1] = 7

    Notes:
    ------
    - The stride parameter allows flexibility in block definition:
      - 'DBM' is suited for non-overlapping blocks, often used in classical block maxima methods.
      - 'SBM' is suited for sliding windows, offering finer granularity for overlap-based analysis.
    - Indexing of blocks and maxima is 1-based for user-friendliness but is internally converted 
      to Python's 0-based indexing.
    """
    if stride == 'DBM':
        return np.sum(data[(j-1)*bs:j*bs] > maxima[i-1])
    if stride == 'SBM':
        return np.sum(data[j:j+bs] > maxima[i-1])

def hat_pi0(data, maxima=None, bs=None, stride='DBM'):
    r"""
    Compute estimators \( \hat{\pi}(1) \) for the limiting cluster size probability \( \pi(1) \).

    This function estimates the first value of \( \pi(m) \), which represent probabilities 
    associated with cluster sizes in stationary time series. 

    Parameters:
    -----------

    data : array-like
        The dataset or time series used to compute probabilities and estimators.

    maxima : array-like, optional
        A pre-computed array of maxima (e.g., block maxima). If not provided, block maxima 
        are computed internally using the specified `bs`.

    bs : int, optional
        The block size, specifying the number of elements in each block. If not provided, it is 
        inferred from the length of `data` divided by the number of maxima, if maxima are provided.

    stride : {'DBM', 'SBM'}, optional
        Defines the block method:
        - 'DBM' (Disjoint Block Method): Non-overlapping blocks are used.
        - 'SBM' (Sliding Block Method): Overlapping sliding blocks are used.
        Default is 'DBM'.

    Returns:
    --------
    float
        Estimated \( \hat{\pi}(1) \) value.

    Raises:
    -------
    ValueError
        If neither `maxima` nor `bs` is provided.

    Example:
    --------
    >>> import numpy as np
    >>> data = np.array([1, 3, 5, 2, 6, 4, 7, 9, 8, 10])
    >>> maxima = np.array([5, 7, 10])
    >>> hat_pis(1, data, bs=3, stride='SBM')
    [1.5, 1.0]  # Using SBM and calculated block maxima

    Notes:
    ------
    - The recursive formula for \( \hat{\pi}(m) \) is given by:
    
    .. math::
        \hat{\pi}(1) = 4 \bar{p}(1)
        \hat{\pi}(m) = 4 \bar{p}(m) - 2 \sum_{k=1}^{m-1} \hat{\pi}(m-k) \bar{p}(k)
    
    -  This ensures that each \( \hat{\pi}(m) \) is built upon the values of smaller \( m \).
    - If `pbars` is not provided, the function computes \( \bar{p}(m) \) using `pbar_dbm_fast` 
      for the DBM stride. Future extensions can integrate SBM support dynamically.
    - The function is optimized for the DBM stride but can handle SBM if provided with `pbars`.

    References:
    -----------
    Bücher, A., & Jennessen, T. (2022). Statistical analysis for stationary time series 
    at extreme levels: New estimators for the limiting cluster size distribution. 
    Stochastic Processes and their Applications, 149, 75-106.
    """
    if maxima is not None:
        bs = len(data) // len(maxima)
        #print(bs)
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
    
from scipy.special import gamma, digamma, polygamma
from scipy.optimize import root_scalar

# Careful! This was redefined from a previous version! Upsilon_now(x) = Upsilon_former(x+1)

def Upsilon(x, rho0):
    r"""
    Compute the \( \Upsilon(x, \rho_0) \) function specified in bias correction.

    Parameters:
    -----------
    x : float
        The parameter \( x \) in the \( \Upsilon(x,  \rho_0) \) function. Typically, \( x \) relates 
        to the scaling or shape parameter in the analysis.

    rho0 : float
        

    Returns:
    --------
    float
        The computed value of \( \Upsilon(x, \rho_0) \).

    """
    return rho0 * gamma(x+2) + (1-rho0) * gamma(x+1)

def Upsilon_derivative(x, rho0):
    r"""
    Compute the derivative of \( \Upsilon(x, \rho_0) \) function specified in bias correction.

    Parameters:
    -----------
    x : float
        The parameter \( x \) in the \( \Upsilon(x,  \rho_0) \) function. Typically, \( x \) relates 
        to the scaling or shape parameter in the analysis.

    rho0 : float
        

    Returns:
    --------
    float
        The computed value of \( \Upsilon'(x, \rho_0) \).


    """
    return rho0 * gamma(x+2) * digamma(x+2) + (1-rho0) * gamma(x+1) * digamma(x+1)

def Upsilon_2ndderivative(x, rho0):
    r"""
    Compute the second derivative \( \Upsilon(x, \rho_0) \) function specified in bias correction.

    Parameters:
    -----------
    x : float
        The parameter \( x \) in the \( \Upsilon(x,  \rho_0) \) function. Typically, \( x \) relates 
        to the scaling or shape parameter in the analysis.

    rho0 : float
        

    Returns:
    --------
    float
        The computed value of \( \Upsilon''(x, \rho_0) \).


    """
    return rho0 * gamma(x+2) * (digamma(x+2)**2+polygamma(1,2+x)) + (1-rho0) * gamma(x+1) * (digamma(x+2)**2+polygamma(1,2+x))

def Pi(x, rho0):
    r"""
    Compute the \( \Pi(x, \rho_0) \) function specified in bias correction, closely related to \(\Upsilon\).

    Parameters:
    -----------
    x : float
        The parameter \( x \) in the \( \Upsilon(x,  \rho_0) \) function. Typically, \( x \) relates 
        to the scaling or shape parameter in the analysis.

    rho0 : float
        

    Returns:
    --------
    float
        The computed value of \( \Pi(x, \rho_0) \).


    """
    return 1/x - Upsilon_derivative(x, rho0)/Upsilon(x, rho0)+rho0/2 - np.euler_gamma # 0.5772156649015328606065120#np.euler_gamma

def Psi(a, a_true, rho0):
    r"""
    Compute the \( \Psi(x, \rho_0) \) function specified in bias correction, closely related to \(\Upsilon\).

    Parameters:
    -----------
    x : float
        The parameter \( x \) in the \( \Upsilon(x,  \rho_0) \) function. Typically, \( x \) relates 
        to the scaling or shape parameter in the analysis.

    rho0 : float
        

    Returns:
    --------
    float
        The computed value of \( \Psi(x, \rho_0) \).


    """
    vp = a/a_true
    term = 1/vp - Upsilon_derivative(vp, rho0)/Upsilon(vp, rho0)+rho0/2 - np.euler_gamma#0.5772156649015328606065120#np.euler_gamma
    return 2/a_true * term

def a1_asy(a_true, rho0):
    r"""
    Compute the numerical root of \( \Psi(x, \rho_0) \).

    Parameters:
    -----------
    a_true : float
        
    rho0 : float
        

    Returns:
    --------
    float
        The computed root of \( \Pi(x, \rho_0) \).


    """
    sol = root_scalar(Psi, args=(a_true, rho0), bracket=[1e-2, 100])
    return sol.root

def varpi(rho0):
    r"""
    Compute the numerical root of \( \Pi(x, \rho_0) \).

    Parameters:
    -----------
    rho0 : float
        

    Returns:
    --------
    float
        The computed root of \( \Pi(x, \rho_0) \).

    """
    sol = root_scalar(Pi, args=(rho0,), bracket=[0.01, 10])
    return sol.root

def z0(rho0):
    r"""
    Compute \(z_0\), a special quantity related to bias-correcting \(\sigma\) \( \Pi(x, \rho_0) \).

    Parameters:
    -----------
    rho0 : float
        

    Returns:
    --------
    float
        The computed root of \( \Pi(x, \rho_0) \).

    """
    return 1/2*Upsilon(1+varpi(rho0), rho0) 