Reference: Miscellaneous
========================

This module is a collection for non-specialized, frequently used or basic functions. 

Overview
--------

The `xtremes.miscellaneous` module provides a variety of utility functions that can be used across different parts of your project. These functions are designed to be general-purpose and can help simplify common tasks.

Basic Functions
---------------

The following functions are basic utility functions provided by the `xtremes.miscellaneous` module:

.. autofunction:: xtremes.miscellaneous.sigmoid
.. autofunction:: xtremes.miscellaneous.invsigmoid
.. autofunction:: xtremes.miscellaneous.mse

Examples
--------

1. **Sigmoid Function**:

    .. code-block:: python

        import numpy as np
        from xtremes.miscellaneous import sigmoid

        x = np.array([-2, -1, 0, 1, 2])
        result = sigmoid(x)
        print("Sigmoid Result:", result)

2. **Inverse Sigmoid Function**:

    .. code-block:: python

        import numpy as np
        from xtremes.miscellaneous import invsigmoid

        y = np.array([0.1, 0.5, 0.9])
        result = invsigmoid(y)
        print("Inverse Sigmoid Result:", result)

3. **Mean Squared Error (MSE)**:

    .. code-block:: python

        from xtremes.miscellaneous import mse

        gammas = [0.1, 0.2, 0.3]
        gamma_true = 0.2
        mse_value, variance, bias = mse(gammas, gamma_true)
        print("MSE:", mse_value, "Variance:", variance, "Bias:", bias)

The GEV and its Likelihood
--------------------------

The following functions are related to the Generalized Extreme Value (GEV) distribution and its likelihood:

.. autofunction:: xtremes.miscellaneous.GEV_pdf
.. autofunction:: xtremes.miscellaneous.GEV_cdf
.. autofunction:: xtremes.miscellaneous.GEV_ll

Examples
--------

1. **GEV CDF**:

    .. code-block:: python

        import numpy as np
        from xtremes.miscellaneous import GEV_cdf

        x = np.array([1, 2, 3, 4, 5])
        result = GEV_cdf(x, gamma=0.5, mu=2, sigma=1, theta=0.8)
        print("GEV CDF Result:", result)

2. **GEV PDF**:

    .. code-block:: python

        import numpy as np
        from xtremes.miscellaneous import GEV_pdf

        x = np.array([1, 2, 3, 4, 5])
        result = GEV_pdf(x, gamma=0.5, mu=2, sigma=1)
        print("GEV PDF Result:", result)

3. **GEV Log-Likelihood**:

    .. code-block:: python

        import numpy as np
        from xtremes.miscellaneous import GEV_ll

        x = np.array([1, 2, 3, 4, 5])
        result = GEV_ll(x, gamma=0.5, mu=2, sigma=1)
        print("GEV Log-Likelihood Result:", result)

Piece Wise Moment Estimation 
----------------------------

The following functions are related to Probability Weighted Moment (PWM) estimation:

.. autofunction:: xtremes.miscellaneous.PWM_estimation
.. autofunction:: xtremes.miscellaneous.PWM2GEV

Examples
--------

1. **PWM Estimation**:

    .. code-block:: python

        import numpy as np
        from xtremes.miscellaneous import PWM_estimation

        maxima = np.array([5, 8, 12, 15, 18])
        result = PWM_estimation(maxima)
        print("PWM Estimation Result:", result)

2. **PWM to GEV**:

    .. code-block:: python

        from xtremes.miscellaneous import PWM2GEV

        b_0 = 10
        b_1 = 20
        b_2 = 30
        result = PWM2GEV(b_0, b_1, b_2)
        print("PWM to GEV Result:", result)

Simulating Time Series
----------------------

The following functions are related to simulating time series data:

.. autofunction:: xtremes.miscellaneous.simulate_timeseries
.. autofunction:: xtremes.miscellaneous.stride2int
.. autofunction:: xtremes.miscellaneous.modelparams2gamma_true

Examples
--------

1. **Simulate Time Series**:

    .. code-block:: python

        from xtremes.miscellaneous import simulate_timeseries

        simulated_ts = simulate_timeseries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.7, seed=42)
        print("Simulated Time Series:", simulated_ts[:10])

2. **Stride to Integer**:

    .. code-block:: python

        from xtremes.miscellaneous import stride2int

        stride = 'DBM'
        block_size = 10
        result = stride2int(stride, block_size)
        print("Stride to Integer Result:", result)

3. **Model Parameters to Gamma True**:

    .. code-block:: python

        from xtremes.miscellaneous import modelparams2gamma_true

        distr = 'GEV'
        correlation = 'IID'
        modelparams = [0.5]
        result = modelparams2gamma_true(distr, correlation, modelparams)
        print("Model Parameters to Gamma True Result:", result)