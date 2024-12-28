Usage
=====

.. _installation:

Installation
------------

To use xtremes, it is possible to install it via pip

.. code-block:: console

   (.venv) $ pip install xtremes


The modules
-----------
So far, three modules are implemented. The module ``xtremes.miscellaneous`` contains basic functionalities, 
whereas ``xtremes.HigherOrderStatistics`` is specialized on the influence of higher order statistics for 
Maximum Likelihood estimations. ``xtremes.Bootstrap`` provides a suitable Bootstrap device
for block maxima.

For each module, there will be a tutorial and a subsection in the API reference.


.. toctree::
   :maxdepth: 1
   :caption: Usage Notebooks

   notebooks/timeseries.ipynb
   notebooks/estimators.ipynb
   notebooks/real_data.ipynb
   notebooks/bootstrap.ipynb