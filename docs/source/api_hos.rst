Reference: HigherOrderStatistics
================================
This module is specialized in analyzing the influence of higher order statistics for Maximum Likelihood estimations. 

Overview
--------

The `xtremes.HigherOrderStatistics` module provides tools for analyzing higher order statistics and their influence on Maximum Likelihood estimations. It includes classes and functions for handling time series data, extracting block maxima, and performing statistical analysis.

Classes
-------

The ``TimeSeries`` Class and its Functionalities
------------------------------------------------

The `TimeSeries` class is used to handle and manipulate time series data. It provides methods for extracting block maxima and high order statistics.

.. autoclass:: xtremes.HigherOrderStatistics.TimeSeries
    :members:
    :undoc-members:
    :show-inheritance:

Examples
--------

1. **Extract Block Maxima**:

    .. code-block:: python

        from xtremes.HigherOrderStatistics import TimeSeries
        import numpy as np

        ts_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        ts = TimeSeries(ts_data)
        block_maxima = ts.extract_BM(block_size=5, stride='DBM')
        print("Block Maxima:", block_maxima)

2. **Extract High Order Statistics**:

    .. code-block:: python

        from xtremes.HigherOrderStatistics import TimeSeries
        import numpy as np

        ts_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        ts = TimeSeries(ts_data)
        high_order_stats = ts.extract_HOS(orderstats=3, block_size=5, stride='DBM')
        print("High Order Statistics:", high_order_stats)

The ``HighOrderStats`` Class and its Functionalities
----------------------------------------------------

The `HighOrderStats` class is used to compute and analyze higher order statistics from the time series data. It provides methods for calculating log-likelihoods and performing Maximum Likelihood Estimation (MLE).

.. autoclass:: xtremes.HigherOrderStatistics.HighOrderStats
    :members:
    :undoc-members:
    :show-inheritance:

Examples
--------

1. **Log Likelihood**:

    .. code-block:: python

        from xtremes.HigherOrderStatistics import HighOrderStats
        import numpy as np

        hos_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.5], [0.4, 0.6]])
        hos = HighOrderStats(hos_data)
        log_likelihood = hos.log_likelihood(gamma=0.5, mu=0, sigma=2, r=2)
        print("Log Likelihood:", log_likelihood)

2. **Frechet Log Likelihood**:

    .. code-block:: python

        from xtremes.HigherOrderStatistics import HighOrderStats
        import numpy as np

        hos_data = np.array([[0.5, 1.0], [1.5, 2.0], [1.2, 2.2], [2.0, 3.0]])
        hos = HighOrderStats(hos_data)
        frechet_log_likelihood = hos.Frechet_log_likelihood(alpha=2, sigma=1.5, r=2)
        print("Frechet Log Likelihood:", frechet_log_likelihood)

The ``Data`` Class and its Functionalities
------------------------------------------

The `Data` class is used to handle and manipulate real data for analysis.

.. autoclass:: xtremes.HigherOrderStatistics.Data
    :members:
    :undoc-members:
    :show-inheritance:

The ``PWM_estimators`` Class
----------------------------

The `PWM_estimators` class is used to compute Probability Weighted Moment (PWM) estimators.

.. autoclass:: xtremes.HigherOrderStatistics.PWM_estimators
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: xtremes.HigherOrderStatistics.automatic_parameter_initialization

The ``ML_estimators``, ``Frechet_ML_estimators`` and ``ML_estimators_data`` Classes
-----------------------------------------------------------------------------------

The `ML_estimators`, `Frechet_ML_estimators`, and `ML_estimators_data` classes are used for performing Maximum Likelihood Estimation (MLE) and handling the results.

.. autoclass:: xtremes.HigherOrderStatistics.ML_estimators
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: xtremes.HigherOrderStatistics.Frechet_ML_estimators
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: xtremes.HigherOrderStatistics.log_likelihood
.. autofunction:: xtremes.HigherOrderStatistics.Frechet_log_likelihood

Running Extensive Simulations
-----------------------------

The `xtremes.HigherOrderStatistics` module also provides functions for running extensive simulations and performing multiple MLEs.

.. autofunction:: xtremes.HigherOrderStatistics.run_ML_estimation
.. autofunction:: xtremes.HigherOrderStatistics.run_multiple_ML_estimations

Examples
--------

1. **Run ML Estimation**:

    .. code-block:: python

        from xtremes.HigherOrderStatistics import run_ML_estimation

        result = run_ML_estimation("timeseries_data.pkl", corr='ARMAX', gamma_true=0.5, block_sizes=[10, 20, 30], stride='DBM', option=2, estimate_pi=True)
        print(result)

2. **Run Multiple ML Estimations**:

    .. code-block:: python

        from xtremes.HigherOrderStatistics import run_multiple_ML_estimations
        import numpy as np

        result = run_multiple_ML_estimations("timeseries_data.pkl", corr='IID', gamma_trues=np.arange(-4, 5, 1)/10, block_sizes=[10, 20, 30], stride='SBM', option=1, estimate_pi=False, parallelize=True)
        print(result)