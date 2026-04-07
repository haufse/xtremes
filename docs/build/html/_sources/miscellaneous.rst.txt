
============================================
Utility Functions for Extreme Value Analysis
============================================

In this tutorial, we will explore the utility functions provided in `miscellaneous.py` that are used in the context of extreme value analysis. These functions include calculating probability-weighted moments, simulating time series, computing GEV distributions, and performing basic statistical tasks like sigmoid and inverse sigmoid transformations.

We will cover:
- Basic mathematical functions like `sigmoid()` and `mse()`.
- Functions related to Generalized Extreme Value (GEV) distributions such as `GEV_pdf()`, `GEV_cdf()`, and `PWM_estimation()`.
- Simulating time series data using the `simulate_timeseries()` function.

Let’s go through each function step by step.

Step 1: Using the Sigmoid and Inverse Sigmoid Functions
=======================================================
The sigmoid function is used to map real-valued numbers into the range (0, 1), and the inverse sigmoid performs the opposite transformation.

Here’s how you can use the `sigmoid()` and `invsigmoid()` functions:

.. code-block:: python

    from miscellaneous import sigmoid, invsigmoid

    # Apply sigmoid transformation
    x = [-2, -1, 0, 1, 2]
    sigmoid_values = sigmoid(x)
    print(sigmoid_values)

    # Apply inverse sigmoid transformation
    y = [0.1, 0.5, 0.9]
    invsigmoid_values = invsigmoid(y)
    print(invsigmoid_values)

The sigmoid function maps any real-valued input to a value between 0 and 1, while the inverse sigmoid transforms probabilities back to their original values.

Step 2: Calculating Probability Weighted Moments (PWM)
======================================================
In extreme value theory, PWMs are used to estimate parameters of the Generalized Extreme Value (GEV) distribution. The function `PWM_estimation()` computes the first three PWMs based on block maxima.

Here’s how to compute PWM for a set of block maxima:

.. code-block:: python

    from miscellaneous import PWM_estimation

    # Example block maxima data
    maxima = [5, 8, 12, 15, 18]

    # Compute PWM estimators
    beta_0, beta_1, beta_2 = PWM_estimation(maxima)
    print(f"β0: {beta_0}, β1: {beta_1}, β2: {beta_2}")

These PWMs can then be used to estimate GEV parameters using the `PWM2GEV()` function, which converts PWMs to GEV parameters (shape, location, and scale).

Step 3: Estimating GEV Parameters from PWM
==========================================
The function `PWM2GEV()` converts the first three PWM moments into GEV distribution parameters: shape (γ), location (μ), and scale (σ).

Here’s how to compute GEV parameters from PWM estimators:

.. code-block:: python

    from miscellaneous import PWM2GEV

    # PWM estimators
    b_0, b_1, b_2 = 11.6, 11.2, 39.2

    # Compute GEV parameters
    gamma, mu, sigma = PWM2GEV(b_0, b_1, b_2)
    print(f"GEV Shape (γ): {gamma}, Location (μ): {mu}, Scale (σ): {sigma}")

The `PWM2GEV()` function allows you to estimate the GEV distribution parameters based on the computed PWM moments.

Step 4: Working with the GEV Distribution
=========================================
The module provides several functions to compute properties of the Generalized Extreme Value (GEV) distribution, including:
- `GEV_pdf()`: Computes the Probability Density Function (PDF).
- `GEV_cdf()`: Computes the Cumulative Distribution Function (CDF).
- `GEV_ll()`: Computes the log-likelihood of the GEV distribution.

Here’s how to use these functions:

.. code-block:: python

    from miscellaneous import GEV_pdf, GEV_cdf, GEV_ll

    # Example data
    x = [1, 2, 3, 4, 5]

    # Compute GEV PDF
    pdf_values = GEV_pdf(x, gamma=0.5, mu=2, sigma=1)
    print("GEV PDF:", pdf_values)

    # Compute GEV CDF
    cdf_values = GEV_cdf(x, gamma=0.5, mu=2, sigma=1)
    print("GEV CDF:", cdf_values)

    # Compute GEV log-likelihood
    ll_values = GEV_ll(x, gamma=0.5, mu=2, sigma=1)
    print("GEV Log-Likelihood:", ll_values)

These functions allow you to work with GEV distributions for various tasks like computing probabilities, densities, or performing likelihood-based inference.

Step 5: Simulating Time Series Data
===================================
The `simulate_timeseries()` function is a powerful utility to generate time series data with different distributions and correlation structures. You can simulate IID (independent and identically distributed) data or time series with temporal dependence using ARMAX or AR models.

Here’s how to simulate a time series:

.. code-block:: python

    from miscellaneous import simulate_timeseries

    # Simulate an IID time series with GEV distribution
    simulated_ts = simulate_timeseries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], seed=42)

    # Print the first 10 values
    print(simulated_ts[:10])

This function supports various distributions (e.g., GEV, Frechet, GPD) and allows you to introduce temporal dependence using ARMAX or AR models.

Conclusion
==========
In this tutorial, we explored several utility functions provided in `miscellaneous.py` for extreme value analysis. These functions help in tasks ranging from basic mathematical transformations (like sigmoid) to more advanced operations like PWM estimation, GEV parameter estimation, and time series simulation.

