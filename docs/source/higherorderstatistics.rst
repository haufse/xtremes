
======================================
Time Series and Extreme Value Analysis
======================================

In this tutorial, we will explore how to use the functionalities provided in `HigherOrderStatistics.py` to simulate time series data, extract block maxima, and perform Maximum Likelihood Estimation (MLE) for extreme value distributions. We will also see how to work with real data to extract high-order statistics and compute MLE.

We will cover:
- Simulating time series data using the `TimeSeries` class.
- Extracting block maxima and high-order statistics from the simulated data.
- Estimating parameters for GEV and Frechet distributions using PWM and MLE.
- Analyzing real-world datasets for extreme value statistics.

Let's walk through each of these steps with code examples.

Step 1: Simulating Time Series Data
===================================
The first step is to simulate a time series dataset using the `TimeSeries` class. You can specify the length of the series, the type of distribution (e.g., GEV), and whether to apply any correlation structure (e.g., IID or ARMAX).

Here’s how you can simulate a GEV-distributed time series:

.. code-block:: python

    from HigherOrderStatistics import TimeSeries

    # Create a time series object with 100 data points, GEV distribution, and IID correlation
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5])

    # Simulate the time series with 10 repetitions
    ts.simulate(rep=10)

    # Print the simulated time series data
    print(ts.values)

Step 2: Extracting Block Maxima
===============================
Once the time series is generated, the next step is to extract block maxima from the data. Block maxima are the largest values within blocks of a certain size in the time series.

Here’s how to extract block maxima with a block size of 5:

.. code-block:: python

    # Extract block maxima with a block size of 5
    ts.get_blockmaxima(block_size=5, stride='DBM')

    # Print the extracted block maxima
    print(ts.blockmaxima)

In this example, the `get_blockmaxima()` function divides the time series into blocks of size 5 and extracts the maximum value from each block. You can adjust the stride (e.g., 'DBM' for disjoint blocks or 'SBM' for sliding blocks).

Step 3: Extracting High Order Statistics
========================================
High-order statistics refer to the second, third, or higher-largest values within a block of data. You can extract these using the `get_HOS()` method.

Here’s how to extract the top 2 largest values from each block:

.. code-block:: python

    # Extract the two highest values from each block
    ts.get_HOS(orderstats=2, block_size=5, stride='DBM')

    # Print the high-order statistics
    print(ts.high_order_stats)

Step 4: Estimating Parameters with PWM
======================================
Once block maxima are extracted, you can estimate the parameters of the Generalized Extreme Value (GEV) distribution using Probability Weighted Moments (PWM). The `PWM_estimators` class handles this.

.. code-block:: python

    from HigherOrderStatistics import PWM_estimators

    # Initialize PWM estimator with the time series data
    pwm = PWM_estimators(ts)

    # Compute PWM estimators for GEV parameters
    pwm.get_PWM_estimation()

    # Print the GEV parameter estimates
    print(pwm.values)

Step 5: Maximum Likelihood Estimation (MLE)
===========================================
To estimate the GEV or Frechet parameters using MLE, you can use the `ML_estimators` class. This method fits the distribution to the block maxima or high-order statistics.

Here’s how to perform MLE for the GEV distribution:

.. code-block:: python

    from HigherOrderStatistics import ML_estimators

    # Initialize MLE estimator with the time series data
    ml = ML_estimators(ts)

    # Perform MLE for the GEV distribution
    ml.get_ML_estimation()

    # Print the MLE results
    print(ml.values)

Step 6: Analyzing Real Data
===========================
You can also work with real-world datasets using the `Data` class. This class allows you to extract block maxima and high-order statistics, and perform MLE on the dataset.

Here’s how to analyze a real dataset:

.. code-block:: python

    from HigherOrderStatistics import Data

    # Initialize the Data class with a real dataset
    data = Data([2.5, 3.1, 1.7, 4.6, 5.3, 2.2, 6.0])

    # Extract block maxima
    data.get_blockmaxima(block_size=2, stride='DBM')

    # Extract high-order statistics
    data.get_HOS(orderstats=2, block_size=2, stride='DBM')

    # Perform MLE on the dataset
    data.get_ML_estimation(FrechetOrGEV='GEV')

    # Print the MLE results
    print(data.ML_estimators.values)

Conclusion
==========
In this tutorial, we explored how to simulate time series data, extract block maxima and high-order statistics, and perform MLE for extreme value distributions. We also saw how to analyze real-world data for extreme value statistics using block maxima and MLE.
