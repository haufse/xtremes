Usage
=====

.. _installation:

Installation
------------

To use xtremes, in near future it will be possible to install it using pip:

.. code-block:: console

   (.venv) $ pip install xtremes

Until now, only a preliminary test version is avaiable:

.. code-block:: console

   (.venv) $ pip install -i https://test.pypi.org/simple/ test-xtremes

Creating stuff
----------------

To be adapted later:
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import xtrenes
>>> xtremes.miscellaneous.sigmoid(1)
0.5
