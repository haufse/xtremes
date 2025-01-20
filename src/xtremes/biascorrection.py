## Implements the bias correction based on Bücher and Jennessen
# only for the Frechet case relevant

import numpy as np
from scipy.special import gamma as sps_gamma
from scipy.special import digamma
from scipy.optimize import root_scalar
from tqdm import tqdm
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
    return np.arange(j, j + bs)

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
    
    return [(i, j) for i in np.arange(1,n-bs+2) 
            for j in np.arange(1,n-bs+2) 
            if np.intersect1d(I_sb(i, bs), I_sb(j, bs)).size == 0]
    
# does the same in fast
def D_n_fast(n, bs):
    r"""
    Efficiently generate all pairs of indices representing disjoint blocks in a time series.

    This function improves upon the basic implementation of `D_n` by using optimized logic 
    to directly construct pairs of disjoint block indices `(i, j)`. Blocks `I_sb(i, bs)` and 
    `I_sb(j, bs)` are considered disjoint if there is no overlap between the indices they cover.

    Parameters:
    -----------
    n : int
        The length of the time series or dataset. This defines the range of valid 
        starting indices for blocks.
        
    bs : int
        The block size, specifying the number of elements in each block. Blocks 
        are defined as contiguous subsets of the time series.

    Returns:
    --------
    list of tuples
        A list of tuples `(i, j)`, where `i` and `j` are the starting indices of 
        two disjoint blocks. The blocks starting at these indices are non-overlapping.

    Example:
    --------
    >>> import numpy as np
    >>> D_n_fast(6, 2)
    [(1, 4), (1, 5), (1, 6), (2, 5), (2, 6), (3, 6), (4, 1), (4, 2), (4, 3),
     (5, 1), (5, 2), (5, 3), (6, 1), (6, 2), (6, 3)]

    Notes:
    ------
    - The function assumes that the helper function `I_sb(i, bs)` defines the range of 
      indices covered by a block starting at `i`.
    - Instead of comparing all pairs `(i, j)` to check for disjointness, it partitions 
      the problem into two subsets:
        1. `j + bs <= i`: Blocks starting before `i` that end before or at the start of `i`.
        2. `j >= i + bs`: Blocks starting after `i` that begin after or at the end of `i`.
    - This reduces unnecessary comparisons, improving computational efficiency.

    - The function uses numpy indexing to quickly find indices satisfying these conditions.

    References:
    -----------
    Bücher, A., & Jennessen, T. (2022). Statistical analysis for stationary time series 
    at extreme levels: New estimators for the limiting cluster size distribution. 
    Stochastic Processes and their Applications, 149, 75-106.
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

def pbar(m, data, maxima=None, bs=None, stride='DBM'):
    r"""
    Estimate the probability of observing exactly `m` exceedances in a block, 
    given maxima and block definitions.

    This function computes an estimator for \( \bar{p}(m) \), which represents the 
    probability of observing exactly `m` exceedances of a reference maximum in a block. 
    It can handle both disjoint and sliding block methods depending on the `stride` parameter.

    Parameters:
    -----------
    m : int
        The number of exceedances to evaluate. This is the target count of values 
        exceeding a reference maximum within a block.

    data : array-like
        The dataset or time series from which maxima and exceedances are calculated.

    maxima : array-like, optional
        A pre-computed array of maxima (e.g., block maxima) from the dataset. 
        If not provided, `bs` must be specified to compute maxima internally.

    bs : int, optional
        The block size, specifying the number of elements in each block. 
        If not provided, it is inferred from the length of `data` divided by the 
        number of maxima, if maxima are provided.

    stride : {'DBM', 'SBM'}, optional
        Defines the block method:
        - 'DBM' (Disjoint Block Method): Non-overlapping blocks are used.
        - 'SBM' (Sliding Block Method): Overlapping sliding blocks are used.
        Default is 'DBM'.

    Returns:
    --------
    float
        The estimated probability \( \bar{p}(m) \), normalized by the total number 
        of valid block pairs.

    Raises:
    -------
    ValueError
        If neither `maxima` nor `bs` is provided.

    Example:
    --------
    >>> import numpy as np
    >>> data = np.array([1, 3, 5, 2, 6, 4, 7, 9, 8, 10])
    >>> maxima = np.array([5, 7, 10])
    >>> pbar(2, data, maxima=maxima, stride='DBM')
    0.16666666666666666  # Probability of exactly 2 exceedances in DBM blocks
    >>> pbar(1, data, bs=3, stride='SBM')
    0.25  # Probability of exactly 1 exceedance in SBM blocks of size 3

    Notes:
    ------
    - The function either requires pre-computed maxima or a block size (`bs`) 
      to compute block maxima internally.
    - In the 'DBM' mode, it iterates over all pairs of disjoint blocks and computes the 
      exceedances, normalized by the total number of such pairs.
    - In the 'SBM' mode, it uses `D_n_fast` to efficiently identify all pairs of 
      disjoint sliding blocks.

    References:
    -----------
    Bücher, A., & Jennessen, T. (2022). Statistical analysis for stationary time series 
    at extreme levels: New estimators for the limiting cluster size distribution. 
    Stochastic Processes and their Applications, 149, 75-106.
    """
    if maxima is not None:
        bs = len(data) // len(maxima)
        # print(bs)
    elif bs is not None:
        maxima = topt.extract_BM(data, bs, stride=stride)
    else:
        raise ValueError('Either maxima or block size must be provided')
    k = len(maxima)
    if stride == 'DBM':
        s = np.sum([exceedances(data, maxima, bs, i, j,stride='DBM')==m
                    for i in range(1,1+k)
                    for j in np.delete(np.arange(1,1+k), i-1)
                    ])
        return  s/(k*(k-1))  
    if stride == 'SBM':
        n = len(data)
        s = np.sum([exceedances(data, maxima, bs, i, j,stride='SBM')==m
                    for (i,j) in D_n_fast(n, bs)
                    ])
        return  s/len(D_n_fast(n, bs))

# does the same in fast
def pbar_dbm_fast(m, data, maxima=None, bs=None):
    r""""
    Compute the estimated probability \( \bar{p}(m) \) for the Disjoint Block Method (DBM) 
    in a highly optimized manner using precomputed cumulative sums.

    This function improves the efficiency of the DBM stride calculation by leveraging 
    vectorized operations and cumulative sums, avoiding nested loops. It computes 
    \( \bar{p}(m) \), which represents the probability of observing exactly `m` exceedances 
    in disjoint blocks of a time series.

    Parameters:
    -----------
    m : int
        The number of exceedances to evaluate. This is the target count of values 
        exceeding a reference maximum within a block.

    data : array-like
        The dataset or time series from which maxima and exceedances are calculated.
        
    maxima : array-like, optional
        A pre-computed array of maxima (e.g., block maxima) from the dataset. If not provided, 
        `bs` must be specified to compute maxima internally.

    bs : int, optional
        The block size, specifying the number of elements in each block. If not provided, 
        it is inferred from the length of `data` divided by the number of maxima, 
        if maxima are provided.

    Returns:
    --------
    float
        The estimated probability \( \bar{p}(m) \), normalized by the total number of valid 
        block pairs for the DBM stride.

    Raises:
    -------
    ValueError
        If neither `maxima` nor `bs` is provided.

    Example:
    --------
    >>> import numpy as np
    >>> data = np.array([1, 3, 5, 2, 6, 4, 7, 9, 8, 10])
    >>> maxima = np.array([5, 7, 10])
    >>> pbar_dbm_fast(2, data, maxima=maxima)
    0.16666666666666666  # Probability of exactly 2 exceedances in DBM blocks
    >>> pbar_dbm_fast(1, data, bs=3)
    0.25  # Probability of exactly 1 exceedance in DBM blocks of size 3

    Notes:
    ------
    1. If `maxima` is not provided, it is computed internally using the specified `bs` 
       via a presumed external function `topt.extract_BM`.
    2. This implementation uses:
       - **Broadcasting**: To efficiently compare all data points with all maxima.
       - **Cumulative sums**: To quickly compute sums within disjoint blocks.
       - **Vectorized filtering**: To exclude diagonal elements where block indices match.
    3. The method is highly optimized and avoids nested loops, making it suitable for 
       large datasets.

    Implementation Details:
    -----------------------
    - Constructs a binary matrix `X` where `X[i, t]` indicates if `data[t] > maxima[i]`.
    - Uses cumulative sums `Y` of `X` to efficiently calculate block sums for all pairs 
      of maxima and blocks.
    - Filters out diagonal elements (where block indices match) and counts occurrences 
      of exactly `m` exceedances.

    References:
    -----------
    Bücher, A., & Jennessen, T. (2022). Statistical analysis for stationary time series 
    at extreme levels: New estimators for the limiting cluster size distribution. 
    Stochastic Processes and their Applications, 149, 75-106.
    """
    # Either maxima is given, or we compute it from data & bs
    if maxima is None:
        if bs is None:
            raise ValueError('Either maxima or block size must be provided')
        else:
            maxima = topt.extract_BM(data, bs, stride='DBM')
    else:
        # If maxima given, deduce bs from data
        bs = len(data) // len(maxima)
    
    k = len(maxima)
    n = len(data)
    
    # 1) Build X: shape (k, n), X[i, t] = 1 if data[t] > maxima[i]
    #    This is a vectorized comparison across i,t
    #    We'll do data in one dimension => shape is (1,n) vs (k,1):
    #    then broadcast the comparison
    # 
    #    But watch out for shape mismatch: we want 
    #    X[i,t] = data[t] > maxima[i].
    #    We can do:
    X = (data[None, :] > maxima[:, None]).astype(int)
    
    # 2) Build Y: shape (k, n+1), cumulative sums of each row in X
    Y = np.zeros((k, n+1), dtype=int)
    np.cumsum(X, axis=1, out=Y[:, 1:])  # store cumsum from col1 onward
    
    # 3) For each i (0-based) and each j != i, the sum of the j-th block is:
    #    sum_{t=j*bs}^{(j+1)*bs - 1} X[i,t]
    #    = Y[i, (j+1)*bs] - Y[i, j*bs].
    #
    #    We can compute these block sums for each row i all at once:
    #    block_sums[i, j] = Y[i, (j+1)*bs] - Y[i, j*bs].
    
    # Make an array of block boundaries from 0, bs, 2bs, ... k*bs
    block_edges = np.arange(0, bs*(k+1), bs)
    
    # block_sums will have shape (k, k) because there are k blocks total
    # block_sums[i, j] = sum_{t in block j} X[i,t]
    block_sums = Y[:, block_edges[1:]] - Y[:, block_edges[:-1]]
    
    # 4) Now we want to count how many times block_sums[i,j] == m for j != i.
    #    Then we divide by k*(k-1).
    
    # Let's do it in a vectorized manner:
    mask_offdiag = np.ones((k, k), dtype=bool)
    np.fill_diagonal(mask_offdiag, False)  # j != i
    
    count = np.sum(block_sums[mask_offdiag] == m)
    
    return count / (k * (k - 1))

 
# def hat_pi(m, data, maxima=None, bs=None, stride='DBM'):
#     if m == 1:
#         return 4*pbar(m, data, maxima, bs, stride)
#     else:
#         return 4*pbar(m, data, maxima, bs, stride) - 2 * np.sum([hat_pi(m-k, data, maxima, bs, stride)*pbar(k, data, maxima, bs, stride) for k in range(1,m)])
 
# def hat_pis(m, data, maxima=None, bs=None, stride='DBM'):
#     hp = [hat_pi(k, data, maxima, bs, stride) for k in range(1,m+1)]
#     return np.array(hp)

def hat_pis(m, data, maxima=None, bs=None, stride='DBM', pbars=None):
    r"""
    Compute recursive estimators \( \hat{\pi}(m) \) for the limiting cluster size distribution.

    This function estimates the first `m` values of \( \pi(m) \), which represent probabilities 
    associated with cluster sizes in stationary time series. The calculations are based on a 
    recursive formula utilizing \( \bar{p}(m) \), either precomputed or dynamically generated.

    Parameters:
    -----------
    m : int
        The maximum cluster size to estimate, corresponding to \( \hat{\pi}(1), \ldots, \hat{\pi}(m) \).

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

    pbars : list of float, optional
        A precomputed list of \( \bar{p}(k) \) values for \( k = 1, \ldots, m \). If not provided, 
        these probabilities are computed dynamically using `pbar_dbm_fast` (for DBM stride).

    Returns:
    --------
    list of float
        A list containing the estimated \( \hat{\pi}(k) \) values for \( k = 1, \ldots, m \).

    Raises:
    -------
    ValueError
        If neither `maxima` nor `bs` is provided.

    Example:
    --------
    >>> import numpy as np
    >>> data = np.array([1, 3, 5, 2, 6, 4, 7, 9, 8, 10])
    >>> maxima = np.array([5, 7, 10])
    >>> hat_pis(3, data, maxima=maxima, stride='DBM')
    [2.0, 1.6, 1.2]  # Example estimated cluster probabilities
    >>> hat_pis(2, data, bs=3, stride='SBM')
    [1.5, 1.0]  # Using SBM and calculated block maxima

    Notes:
    ------
    - The recursive formula for \( \hat{\pi}(m) \) is given by:
      \[
      \hat{\pi}(1) = 4 \bar{p}(1)
      \]
      \[
      \hat{\pi}(m) = 4 \bar{p}(m) - 2 \sum_{k=1}^{m-1} \hat{\pi}(m-k) \bar{p}(k)
      \]
      This ensures that each \( \hat{\pi}(m) \) is built upon the values of smaller \( m \).
    - If `pbars` is not provided, the function computes \( \bar{p}(m) \) using `pbar_dbm_fast` 
      for the DBM stride. Future extensions can integrate SBM support dynamically.
    - The function is optimized for the DBM stride but can handle SBM if provided with `pbars`.

    References:
    -----------
    Bücher, A., & Jennessen, T. (2022). Statistical analysis for stationary time series 
    at extreme levels: New estimators for the limiting cluster size distribution. 
    Stochastic Processes and their Applications, 149, 75-106.
    """
    if pbars == None:
        #print('\r Computing pbars',end='')
        #pbars = [pbar(k, data, maxima, bs, stride) for k in range(1,m+1)]
        if stride=='DBM':
            pbars = [pbar_dbm_fast(k, data, maxima, bs) for k in range(1,m+1)]
        #print('\r pbars computed  ',end='')

    hps = [4*pbars[0]]
    if m == 1:
        return hps
    else:
        #for q in tqdm(range(1,m)):
        for q in range(1,m):
            hps.append(4*pbars[q]-2*np.sum([hps[-k]*pbars[k] for k in range(1,q)]))
    return hps
    


def convolve_probabilities(pi, l, k):
    r"""
    Compute the probability \( P \left( \sum_{i=1}^l K_i = k \right) \) using iterative convolutions.

    This function calculates the probability of obtaining a sum of `k` from `l` independent random 
    variables \( K_1, K_2, \ldots, K_l \) that share the same probability distribution `pi`. 
    The result is determined by convolving the probability distribution `pi` iteratively `l` times.

    Parameters:
    -----------
    pi : numpy.ndarray
        An array of probabilities \( [\pi(1), \pi(2), \ldots, \pi(r)] \), where \( \pi(i) \) represents 
        the probability of the random variable \( K \) taking the value \( i \). Indexing assumes 
        0-based arrays, so `pi[k]` corresponds to \( \pi(k) \).

    l : int
        The number of independent random variables in the sum. This represents the number of convolutions 
        to perform.

    k : int
        The desired value of the sum \( \sum_{i=1}^l K_i = k \) for which the probability is computed.

    Returns:
    --------
    float
        The probability \( P \left( \sum_{i=1}^l K_i = k \right) \). If `k` exceeds the maximum possible 
        value, the function returns 0.0.

    Example:
    --------
    >>> import numpy as np
    >>> pi = np.array([0.2, 0.5, 0.3])  # P(K=1)=0.2, P(K=2)=0.5, P(K=3)=0.3
    >>> convolve_probabilities(pi, l=2, k=3)
    0.25  # Probability of K1 + K2 = 3
    >>> convolve_probabilities(pi, l=3, k=5)
    0.17  # Probability of K1 + K2 + K3 = 5

    Notes:
    ------
    - The convolution of two probability distributions calculates the probability distribution of the 
      sum of two independent random variables.
    - The function uses numpy's `np.convolve`, which performs discrete, linear convolution. 
      After `l-1` convolutions, the resulting array represents the probability distribution of 
      \( \sum_{i=1}^l K_i \).
    - The probability for a specific `k` is retrieved from the resulting array.

    Edge Cases:
    -----------
    - If `l=1`, the function directly returns \( \pi(k) \) (or 0.0 if \( k > \text{len(pi)} \)).
    - If \( k \) exceeds the sum of maximum possible values, the function returns 0.0.
    """
    if l == 1:
        # Base case: first convolution is just pi
        return pi[k] if k <= len(pi) else 0.0

    # Start with the first distribution
    result = pi.copy()

    # Perform (l - 1) convolutions
    for _ in range(1, l):
        result = np.convolve(result, pi)

    # Retrieve the desired probability
    return result[k] if k - 1 < len(result) else 0.0

def hat_frakp(pi, r, l):
    r"""
    Compute probabilities of the cluster size distribution of exceedances.

    This function calculates the probability associated with cluster sizes for exceedances 
    in stationary time series. It uses a formula involving convolutions of the probability 
    distribution `pi`, summing over possible exceedance counts and normalizing by the 
    gamma function.

    Parameters:
    -----------
    pi : numpy.ndarray
        An array of probabilities \( [\pi(1), \pi(2), \ldots, \pi(r)] \), where \( \pi(i) \) 
        represents the probability of the random variable \( K \) taking the value \( i \).
        These probabilities correspond to the cluster size distribution.

    r : int
        The upper limit of the sum over exceedance counts. It defines the range 
        of possible exceedance values to consider.

    l : int
        The desired cluster size index for which the probability is computed.

    Returns:
    --------
    float
        The probability \( \hat{\mathfrak{p}}_l \), associated with the cluster size `l`.

    Example:
    --------
    >>> import numpy as np
    >>> from scipy.special import gamma as sps_gamma
    >>> pi = np.array([0.2, 0.5, 0.3])  # P(K=1)=0.2, P(K=2)=0.5, P(K=3)=0.3
    >>> hat_frakp(pi, r=5, l=2)
    -0.2  # Example result (illustrative)

    Notes:
    ------
    - The formula for \( \hat{\mathfrak{p}}_l \) is:
      \[
      \hat{\mathfrak{p}}_l = \frac{(-1)^l}{\Gamma(l+1)} \sum_{k=l}^r P \left( \sum_{i=1}^l K_i = k \right)
      \]
      where \( P(\sum_{i=1}^l K_i = k) \) is computed via convolutions of `pi`.
    - The gamma function \( \Gamma(l+1) \) ensures proper normalization, and the alternating 
      sign \( (-1)^l \) introduces corrections based on the cluster size index.

    - The summation \( \sum_{k=l}^r \) considers all valid exceedance counts, starting at `l` 
      (minimum size) and going up to `r` (maximum possible size).

    Requirements:
    -------------
    - The function relies on `convolve_probabilities` to compute the distribution of sums of `l` 
      independent random variables with distribution `pi`.
    - The `sps_gamma` function from `scipy.special` provides the gamma function.

    Edge Cases:
    -----------
    - If `pi` has insufficient length relative to `r`, some probabilities may be treated as zero.
    - Negative or zero values for `l` or `r` will likely result in nonsensical computations.
    """
    return (-1)**l/sps_gamma(l+1)*np.sum([convolve_probabilities(pi, l, k) for k in range(l, r)])

def ups(x, r, pi):
    r"""
    Compute the \( \Upsilon(x, r, \pi) \) function for a given probability distribution \( \pi \).

    The function \( \Upsilon(x, r, \pi) \) is defined as a weighted sum of terms involving 
    \( \hat{\mathfrak{p}}(l) \), gamma functions, and alternating signs. It is used in the 
    context of cluster size distributions in stationary time series analysis.

    Parameters:
    -----------
    x : float
        The parameter \( x \) in the \( \Upsilon(x, r, \pi) \) function. Typically, \( x \) relates 
        to the scaling or shape parameter in the analysis.

    r : int
        The range of terms to sum over. This corresponds to the maximum cluster size 
        considered in the computation.

    pi : numpy.ndarray
        An array of probabilities \( [\pi(1), \pi(2), \ldots, \pi(r)] \), representing the 
        cluster size distribution.

    Returns:
    --------
    float
        The computed value of \( \Upsilon(x, r, \pi) \).

    Example:
    --------
    >>> import numpy as np
    >>> from scipy.special import gamma as sps_gamma
    >>> pi = np.array([0.2, 0.5, 0.3])  # P(K=1)=0.2, P(K=2)=0.5, P(K=3)=0.3
    >>> ups(0.5, r=3, pi=pi)
    -0.125  # Example result (illustrative)

    Notes:
    ------
    - The formula for \( \Upsilon(x, r, \pi) \) is:
      \[
      \Upsilon(x, r, \pi) = \sum_{l=0}^{r-1} \hat{\mathfrak{p}}(l) \cdot (-1)^l \cdot \left( x \cdot \Gamma(l + x) \right)
      \]
      where:
      - \( \hat{\mathfrak{p}}(l) \) is the cluster size probability, computed via `hat_frakp`.
      - The term \( x \cdot \Gamma(l + x) \) is omitted when \( x + l = 0 \) (to handle edge cases).

    - This function extends the recursive cluster size probability computation (`hat_frakp`) 
      into a weighted summation with contributions from \( \pi \).

    Edge Cases:
    -----------
    - If \( x + l = 0 \), the term involving \( \Gamma(l + x) \) is skipped, and only 
      \( \hat{\mathfrak{p}}(l) \) contributes to the sum.
    - Negative or zero values for \( x \) or \( r \) may result in nonsensical computations.

    Dependencies:
    -------------
    - The function `hat_frakp` is used to compute \( \hat{\mathfrak{p}}(l) \).
    - The gamma function \( \Gamma(l + x) \) is provided by `scipy.special.gamma`.
    """
    return np.sum([hat_frakp(pi, r, l)*(-1)**l *x*sps_gamma(l+x) if x+l!=0 else hat_frakp(pi, r, l)*(-1)**l  for l in range(r)])

def ups_d(x, r, pi):
    r"""
    Compute the derivative of \( \Upsilon(x, r, \pi) \) with respect to \( x \).

    The derivative of \( \Upsilon(x, r, \pi) \) incorporates the gamma function, the digamma function, 
    and additional terms for handling edge cases when \( x + l = 0 \). This function is used in the 
    context of cluster size distributions and their analysis.

    Parameters:
    -----------
    x : float
        The parameter \( x \), representing the variable of differentiation in the \( \Upsilon(x, r, \pi) \) function.

    r : int
        The range of terms to sum over. This corresponds to the maximum cluster size 
        considered in the computation.

    pi : numpy.ndarray
        An array of probabilities \( [\pi(1), \pi(2), \ldots, \pi(r)] \), representing the 
        cluster size distribution.

    Returns:
    --------
    float
        The computed derivative of \( \Upsilon(x, r, \pi) \) with respect to \( x \).

    Example:
    --------
    >>> import numpy as np
    >>> from scipy.special import gamma as sps_gamma, digamma
    >>> pi = np.array([0.2, 0.5, 0.3])  # P(K=1)=0.2, P(K=2)=0.5, P(K=3)=0.3
    >>> ups_d(0.5, r=3, pi=pi)
    -0.085  # Example result (illustrative)

    Notes:
    ------
    - The formula for the derivative of \( \Upsilon(x, r, \pi) \) is:
      \[
      \Upsilon'(x, r, \pi) = \sum_{l=0}^{r-1} \hat{\mathfrak{p}}(l) \cdot (-1)^l \cdot 
      \Gamma(l + x) \cdot \left( x \cdot \psi(x + l) + 1 \right)
      \]
      where:
      - \( \hat{\mathfrak{p}}(l) \) is the cluster size probability, computed via `hat_frakp`.
      - \( \psi \) is the digamma function, representing the logarithmic derivative of the gamma function.
    - Special handling is applied when \( x + l = 0 \). In this case, the term reduces to:
      \[
      \hat{\mathfrak{p}}(l) \cdot (-1)^l \cdot (-\gamma),
      \]
      where \( \gamma \) is the Euler-Mascheroni constant.

    Edge Cases:
    -----------
    - If \( x + l = 0 \), the function avoids computing \( \Gamma(l + x) \cdot \psi(x + l) \) 
      and substitutes the term \( -\gamma \) (Euler-Mascheroni constant).

    Dependencies:
    -------------
    - The function `hat_frakp` computes \( \hat{\mathfrak{p}}(l) \).
    - Gamma and digamma functions are provided by `scipy.special.gamma` and `scipy.special.digamma`.
    """
    return np.sum([hat_frakp(pi, r, l)*(-1)**l *sps_gamma(l+x) * (x*digamma(x+l)+1) if x+l!=0 else hat_frakp(pi, r, l)*(-1)**l *(-np.euler_gamma) for l in range(r)])

def Pi(x,r,pi):
    r"""
    Compute the \( \Pi(x, r, \pi) \) function, whose root is used for bias correction of the Maximum Likelihood Estimator (MLE).

    The function \( \Pi(x, r, \pi) \) is defined in terms of the functions \( \Upsilon(x, r, \pi) \) and its derivative \( \Upsilon'(x, r, \pi) \). 
    It incorporates a correction term that sums contributions from \( \Upsilon'(0, j, \pi) \) for \( j = 1, \ldots, r \), scaled by the range \( r \). 
    The root of this function is used to adjust bias in MLE computations for statistical analysis.

    Parameters:
    -----------
    x : float
        The parameter \( x \), representing the variable for which the root of \( \Pi(x, r, \pi) \) is sought.

    r : int
        The range of terms to sum over, representing the maximum cluster size considered in the computation.

    pi : numpy.ndarray
        An array of probabilities \( [\pi(1), \pi(2), \ldots, \pi(r)] \), representing the cluster size distribution.

    Returns:
    --------
    float
        The computed value of \( \Pi(x, r, \pi) \).

    Example:
    --------
    >>> import numpy as np
    >>> from scipy.special import gamma as sps_gamma, digamma
    >>> pi = np.array([0.2, 0.5, 0.3])  # P(K=1)=0.2, P(K=2)=0.5, P(K=3)=0.3
    >>> Pi(0.5, r=3, pi=pi)
    0.345  # Example result (illustrative)

    Formula:
    --------
    The function is computed as:
    \[
    \Pi(x, r, \pi) = \frac{1}{x} - \frac{\Upsilon'(x, r, \pi)}{\Upsilon(x, r, \pi)} + \frac{1}{r} \sum_{j=1}^r \Upsilon'(0, j, \pi)
    \]
    where:
    - \( \Upsilon(x, r, \pi) \) is computed using the `ups` function.
    - \( \Upsilon'(x, r, \pi) \) is computed using the `ups_d` function.

    Notes:
    ------
    - The term \( \frac{1}{x} \) introduces a singularity at \( x = 0 \), so care should be taken when \( x \) is near zero.
    - The final summation term involves \( \Upsilon'(0, j, \pi) \), which is computed for \( j = 1, \ldots, r \).

    Edge Cases:
    -----------
    - If \( x = 0 \), the term \( 1/x \) results in a singularity. Users should ensure \( x > 0 \) for valid computations.
    - If \( \Upsilon(x, r, \pi) = 0 \), the division by \( \Upsilon(x, r, \pi) \) leads to undefined behavior.

    Dependencies:
    -------------
    - `ups(x, r, pi)` computes \( \Upsilon(x, r, \pi) \).
    - `ups_d(x, r, pi)` computes \( \Upsilon'(x, r, \pi) \).
    """
    return 1/x - ups_d(x, r, pi)/ups(x, r, pi) +1/r*np.sum([ups_d(0,j,pi) for j in range(1,r+1)])

# def varpi(r,pi):
#     sol = root_scalar(Pi, args=(r,pi,), bracket=[0.001, 100])
#     return sol.root

def varpi(r, pi):
    r"""
    Solve for \( x \) in the range \([0.001, 100]\) such that \( \Pi(x, r, \pi) = 0 \).

    This function attempts to find a root of the \( \Pi(x, r, \pi) \) function using numerical root-finding methods. 
    If a root cannot be found within the specified range, it returns \( x = 100 \) as a fallback.

    Parameters:
    -----------
    r : int
        The range parameter, representing the maximum cluster size considered.

    pi : numpy.ndarray
        An array of probabilities \( [\pi(1), \pi(2), \ldots, \pi(r)] \), representing the cluster size distribution.

    Returns:
    --------
    float
        The value of \( x \) such that \( \Pi(x, r, \pi) = 0 \), or \( x = 100 \) if no root is found.

    Example:
    --------
    >>> import numpy as np
    >>> from scipy.optimize import root_scalar
    >>> pi = np.array([0.2, 0.5, 0.3])  # Example probabilities
    >>> varpi(3, pi)
    0.345  # Example result (illustrative)

    Notes:
    ------
    - The function \( \Pi(x, r, \pi) \) is wrapped into a single-argument function for use with 
      `scipy.optimize.root_scalar`.
    - The root-finding process attempts to locate \( x \) in the bracket \([0.001, 20]\) 
      where \( \Pi(x, r, \pi) = 0 \).

    Edge Cases:
    -----------
    - If no root is found within the specified range, the function defaults to returning \( x = 100 \).
    - If \( \Pi(x, r, \pi) \) is undefined or exhibits singularities (e.g., when \( x = 0 \)), 
      root-finding may fail.

    Dependencies:
    -------------
    - The function `Pi(x, r, pi)` is required to compute the root.
    """
    # Wrap your function Pi in a single-argument function for root_scalar.
    def objective(x):
        return Pi(x, r, pi)

    # First, try root finding:
    try:
        sol = root_scalar(objective, bracket=[0.001, 20])
        # If it converged, great!
        if sol.converged:
            return sol.root
    except Exception:
        pass
    return 100