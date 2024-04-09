API
===

Miscellaneous
-------------

This module is a collection for non-specialized, frequently used or basic functions. 


Basic functions
^^^^^^^^^^^^^^^

.. autofunction:: test_xtremes.miscellaneous.sigmoid
.. autofunction:: test_xtremes.miscellaneous.invsigmoid
.. autofunction:: test_xtremes.miscellaneous.mse

The GEV and its Likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: test_xtremes.miscellaneous.gev
.. autofunction:: test_xtremes.miscellaneous.GEV
.. autofunction:: test_xtremes.miscellaneous.ll_gev
.. autofunction:: test_xtremes.miscellaneous.ll_GEV

Piece Wise Moment Estimation 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: test_xtremes.miscellaneous.PWM_estimation
.. autofunction:: test_xtremes.miscellaneous.PWM2GEV

Simulating Time Series
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: test_xtremes.miscellaneous.simulate_timeseries
.. autofunction:: test_xtremes.miscellaneous.stride2int
.. autofunction:: test_xtremes.miscellaneous.modelparams2gamma_true


HigherOrderStatistics
---------------------
This module is a specialized on analyzing the influence of higher order statistics for Maximum Likelihood estimations. 


The ``TimeSeries`` Class and its Functionalities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: test_xtremes.HigherOrderStatistics.TimeSeries
.. autofunction:: test_xtremes.HigherOrderStatistics.extract_BM
.. autofunction:: test_xtremes.HigherOrderStatistics.extract_HOS

The ``HighOrderStats`` Class and its Functionalities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: test_xtremes.HigherOrderStatistics.HighOrderStats


The ``ML_estimators`` and ``PWM_estimators`` Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: test_xtremes.HigherOrderStatistics.PWM_estimators
.. autoclass:: test_xtremes.HigherOrderStatistics.ML_estimators
.. autofunction:: test_xtremes.HigherOrderStatistics.automatic_parameter_initialization
.. autofunction:: test_xtremes.HigherOrderStatistics.cost_function

Running Extensive Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: test_xtremes.HigherOrderStatistics.background
.. autofunction:: test_xtremes.HigherOrderStatistics.run_ML_estimation
.. autofunction:: test_xtremes.HigherOrderStatistics.run_multiple_ML_estimations
