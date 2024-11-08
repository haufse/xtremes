Reference: Bootstrap
====================
This module computes a Bootstrap procedure on disjoint or sliding block maxima. 

Overview
--------

The `xtremes.bootstrap` module provides tools for performing bootstrap procedures on block maxima. It includes classes and functions for extracting block maxima, resampling, and estimating parameters using Maximum Likelihood Estimation (MLE).

Classes
-------

The ``FullBootstrap`` Class 
---------------------------
.. autoclass:: xtremes.bootstrap.FullBootstrap
    :members:
    :undoc-members:
    :show-inheritance:



Functions
---------

The ``circmax`` Function
------------------------
.. autofunction:: xtremes.bootstrap.circmax

The ``uniquening`` Function
---------------------------
.. autofunction:: xtremes.bootstrap.uniquening

The ``Bootstrap`` Function
--------------------------
.. autofunction:: xtremes.bootstrap.Bootstrap

The ``aggregate_boot`` Function
-------------------------------
.. autofunction:: xtremes.bootstrap.aggregate_boot

The ``bootstrap_worker`` Function
-------------------------------
.. autofunction:: xtremes.bootstrap.bootstrap_worker

Examples
--------

Here are some examples of how to use the `xtremes.bootstrap` module:

1. **FullBootstrap Class**:

    .. code-block:: python

        import numpy as np
        from xtremes.bootstrap import FullBootstrap

        sample = np.random.rand(100)
        bootstrap = FullBootstrap(sample, bs=10, stride='DBM', dist_type='Frechet')
        bootstrap.run_bootstrap(num_bootstraps=100)
        print("Mean of bootstrap estimates:", bootstrap.statistics['mean'])
        print("Standard deviation of bootstrap estimates:", bootstrap.statistics['std'])



2. **circmax Function**:

    .. code-block:: python

        import numpy as np
        from xtremes.bootstrap import circmax

        sample = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        block_maxima = circmax(sample, bs=5, stride='DBM')
        print("Block Maxima (DBM):", block_maxima)

        block_maxima = circmax(sample, bs=3, stride='SBM')
        print("Block Maxima (SBM):", block_maxima)

3. **uniquening Function**:

    .. code-block:: python

        import numpy as np
        from xtremes.bootstrap import uniquening

        circmaxs = np.array([[1, 2, 2, 3], [2, 3, 3, 4]])
        unique_values = uniquening(circmaxs)
        print("Unique values and counts:", unique_values)

4. **Bootstrap Function**:

    .. code-block:: python

        from xtremes.bootstrap import Bootstrap

        sample = [1, 2, 3, 4, 5]
        bootstrap_sample = Bootstrap(sample)
        print("Bootstrap sample:", bootstrap_sample)

5. **aggregate_boot Function**:

    .. code-block:: python

        import numpy as np
        from xtremes.bootstrap import aggregate_boot

        boot_samp = [(np.array([1, 2, 3]), np.array([1, 1, 2])), (np.array([2, 3]), np.array([2, 1]))]
        aggregated_counts = aggregate_boot(boot_samp)
        print("Aggregated counts:", aggregated_counts)

References
----------

- Bücher, A., & Staud, T. (2024). Bootstrapping Estimators based on the Block Maxima Method. arXiv preprint arXiv:2409.05529.