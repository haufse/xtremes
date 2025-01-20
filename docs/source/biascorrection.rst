===============================
Bias Correction Tutorial
===============================

In this tutorial, we will explore how to work with the `biascorrection` module to estimate cluster size distributions and apply bias corrections to Maximum Likelihood Estimation (MLE).

We will cover:
- Computing probabilities of exceedances with `pbar` and `pbar_dbm_fast`.
- Estimating cluster size probabilities with `hat_pis`.
- Using the \( \Upsilon \) and \( \Pi \) functions for bias correction.
- Solving for bias-corrected parameters using `varpi`.

Let's walk through each of these steps with code examples.

Step 1: Computing Probabilities of Exceedances
==============================================
The first step in analyzing cluster sizes is to compute the probabilities of exceedances in disjoint or sliding blocks. The `pbar` function (or its optimized variant `pbar_dbm_fast`) calculates these probabilities efficiently.

Here's how to compute the exceedance probabilities:

.. code-block:: python

    import numpy as np
    from biascorrection import pbar, pbar_dbm_fast

    # Example dataset
    data = np.random.rand(100)
    block_maxima = np.array([0.8, 0.9, 1.0])  # Example block maxima
    block_size = 10

    # Compute exceedance probabilities with pbar
    probabilities = [pbar(m, data, maxima=block_maxima, stride='DBM') for m in range(1, 4)]
    print("Exceedance Probabilities (pbar):", probabilities)

    # Alternatively, use the optimized version
    probabilities_fast = [pbar_dbm_fast(m, data, maxima=block_maxima, bs=block_size) for m in range(1, 4)]
    print("Exceedance Probabilities (pbar_dbm_fast):", probabilities_fast)

These functions allow you to calculate the probabilities \( \bar{p}(m) \) for different exceedance levels \( m \).

Step 2: Estimating Cluster Size Probabilities
=============================================
Using the exceedance probabilities, we can estimate the probabilities of cluster sizes \( \pi(m) \) with the `hat_pis` function.

.. code-block:: python

    from biascorrection import hat_pis

    # Estimate cluster size probabilities
    cluster_probs = hat_pis(3, data, maxima=block_maxima, stride='DBM')
    print("Cluster Size Probabilities (hat_pis):", cluster_probs)

The `hat_pis` function computes \( \hat{\pi}(m) \) recursively based on \( \bar{p}(m) \).

Step 3: Using the Upsilon Function
==================================
The \( \Upsilon \) function computes weighted sums of cluster size probabilities and is a key component in bias correction.

.. code-block:: python

    from biascorrection import ups

    # Compute the Upsilon function for a given x
    x = 0.5
    r = 3
    upsilon = ups(x, r, cluster_probs)
    print("Upsilon Function Value (ups):", upsilon)

Step 4: Using the Pi Function for Bias Correction
=================================================
The \( \Pi(x) \) function is used to determine the bias-corrected parameter. It combines the \( \Upsilon \) function and its derivative \( \Upsilon'(x) \).

.. code-block:: python

    from biascorrection import Pi

    # Compute the Pi function for a given x
    pi_value = Pi(x, r, cluster_probs)
    print("Pi Function Value (Pi):", pi_value)

Step 5: Solving for Bias-Corrected Parameters
=============================================
Finally, we can solve for the bias-corrected parameter \( x \) by finding the root of \( \Pi(x) \) using the `varpi` function.

.. code-block:: python

    from biascorrection import varpi

    # Solve for the bias-corrected parameter
    bias_corrected_param = varpi(r, cluster_probs)
    print("Bias-Corrected Parameter (varpi):", bias_corrected_param)

The `varpi` function attempts to find a root for \( \Pi(x) \) in the range \([0.001, 20]\). If no root is found, it returns a default value of 100.

Conclusion
==========
In this tutorial, we explored how to compute exceedance probabilities, estimate cluster size probabilities, and use the \( \Upsilon \) and \( \Pi \) functions for bias correction. We also demonstrated how to solve for bias-corrected parameters using `varpi`. These tools are essential for analyzing cluster size distributions and addressing bias in MLE parameter estimates.
