Bias Correction
===============
This module computes bias corrections for the Maximum Likelihood Estimation (MLE) of cluster size distributions in stationary time series. So far, it only works for $t=2$.

Overview
--------

The `biascorrection` module provides tools for estimating probabilities of cluster sizes, calculating related functions like \( \Upsilon \) and \( \Pi \), and finding roots for bias correction.

Functions
---------

The ``I_sb`` Function
----------------------
.. autofunction:: xtremes.biascorrection.I_sb

The ``D_n`` Function
------------------------------
.. autofunction:: xtremes.biascorrection.D_n

The ``exceedances`` Function
-----------------------------
.. autofunction:: xtremes.biascorrection.exceedances

The ``hat_pi0`` Function
---------------------------------------
.. autofunction:: xtremes.biascorrection.hat_pi0

The ``Upsilon`` Function
-------------------------
.. autofunction:: xtremes.biascorrection.Upsilon

The ``Upsilon_derivative`` Function
----------------------
.. autofunction:: xtremes.biascorrection.Upsilon_derivative

The ``Upsilon_2ndderivative`` Function
---------------------------------------
.. autofunction:: xtremes.biascorrection.Upsilon_2ndderivative

The ``Psi`` Function
----------------------
.. autofunction:: xtremes.biascorrection.Psi

The ``a1_asy`` Function
-----------------------
.. autofunction:: xtremes.biascorrection.a1_asy

The ``varpi`` Function
----------------------
.. autofunction:: xtremes.biascorrection.varpi


The ``z0`` Function
----------------------
.. autofunction:: xtremes.biascorrection.z0 
