
==========================
Bootstrap and MLE Tutorial
==========================

In this tutorial, we will explore how to work with block maxima extraction, bootstrap resampling, and Maximum Likelihood Estimation (MLE) for extreme value distributions using the `bootstrap.py` code.

We will cover:
- Block maxima extraction using the `circmax()` function.
- Generating bootstrap samples with `Bootstrap()`.
- Aggregating bootstrap samples and estimating Fréchet and GEV parameters using `ML_Estimator`.
- Running full bootstrapping procedures for MLE with `FullBootstrap`.

Let's walk through each of these steps with code examples.

Step 1: Block Maxima Extraction
===============================
The first step in working with extreme value statistics is often to extract block maxima from a dataset. The `circmax()` function allows you to do this using either disjoint blocks or sliding blocks.

Here's how you can extract block maxima from a sample:

.. code-block:: python

    import numpy as np
    from bootstrap import circmax

    # Example dataset
    sample = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Extract disjoint block maxima (DBM)
    block_maxima_dbm = circmax(sample, bs=5, stride='DBM')
    print("Disjoint Block Maxima (DBM):", block_maxima_dbm)

    # Extract sliding block maxima (SBM)
    block_maxima_sbm = circmax(sample, bs=3, stride='SBM')
    print("Sliding Block Maxima (SBM):", block_maxima_sbm)

As you can see, `circmax()` allows you to specify the block size (`bs`) and the stride method (`DBM` or `SBM`). DBM extracts maxima from non-overlapping blocks, while SBM uses overlapping blocks, providing more block maxima.

Step 2: Bootstrapping a Sample
==============================
Next, we'll generate a bootstrap sample. Bootstrapping is a resampling technique used to estimate the variability of a statistic by randomly resampling the data with replacement.

Here’s how you can generate a bootstrap sample from the block maxima we extracted earlier:

.. code-block:: python

    from bootstrap import Bootstrap

    # Generate a bootstrap sample from the block maxima
    boot_sample = Bootstrap(block_maxima_dbm)
    print("Bootstrap Sample:", boot_sample)

In this case, the `Bootstrap()` function takes a list or array as input and returns a new sample of the same size, created by randomly selecting elements from the original data with replacement.

Step 3: Aggregating the Bootstrap Sample
========================================
Once you have a bootstrap sample, the next step is to aggregate the counts of unique values in the sample. This aggregation helps us prepare the data for MLE by summarizing the frequencies of the unique values.

Here’s how you can aggregate the bootstrap sample:

.. code-block:: python

    from bootstrap import aggregate_boot

    # Aggregate the counts of unique values in the bootstrap sample
    aggregated_sample = aggregate_boot(boot_sample)
    print("Aggregated Bootstrap Sample:", aggregated_sample)

Now, we have an aggregated sample that shows the unique values and their corresponding counts. This aggregated data will be used in the next step for MLE.

Step 4: Maximum Likelihood Estimation (MLE)
===========================================
The `ML_Estimator` class allows us to perform MLE for either the Fréchet or GEV distribution using the aggregated bootstrap sample.

Let’s first perform MLE for the Fréchet distribution:

.. code-block:: python

    from bootstrap import ML_Estimator

    # Initialize the ML_Estimator with the aggregated bootstrap sample
    estimator = ML_Estimator(aggregated_sample)

    # Perform MLE for the Fréchet distribution
    frechet_params = estimator.maximize_frechet()
    print("Estimated Fréchet Parameters:", frechet_params)

Similarly, you can perform MLE for the GEV distribution:

.. code-block:: python

    # Perform MLE for the GEV distribution
    gev_params = estimator.maximize_gev()
    print("Estimated GEV Parameters:", gev_params)

With these methods, you can estimate the parameters (shape, scale, location) of both the Fréchet and GEV distributions using the MLE approach.

Step 5: Running Full Bootstrap for MLE
======================================
Finally, to estimate the variability of the MLE parameters, we can use the `FullBootstrap` class. This class applies the full bootstrapping procedure, including resampling, block maxima extraction, and MLE estimation, to obtain mean and standard deviation estimates for the parameters.

Here’s how to run the full bootstrap procedure for the Fréchet distribution:

.. code-block:: python

    from bootstrap import FullBootstrap

    # Example dataset
    sample = np.random.rand(100)

    # Initialize FullBootstrap with the dataset
    bootstrap = FullBootstrap(sample, bs=10, stride='DBM', dist_type='Frechet')

    # Run the bootstrap procedure
    bootstrap.run_bootstrap(num_bootstraps=100)

    # Print the mean and standard deviation of the estimates
    print("Mean of Bootstrap Estimates:", bootstrap.statistics['mean'])
    print("Standard Deviation of Bootstrap Estimates:", bootstrap.statistics['std'])

This process generates multiple bootstrap samples, applies MLE to each, and calculates the mean and standard deviation of the resulting estimates. You can also do this for the GEV distribution by setting `dist_type='GEV'`.

Conclusion
==========
In this tutorial, we walked through the process of extracting block maxima, generating bootstrap samples, aggregating the data, and estimating parameters using MLE. We also saw how to apply the full bootstrap procedure to analyze the variability of the MLE estimates. This framework is essential when dealing with extreme value theory and understanding the uncertainty in parameter estimates.

