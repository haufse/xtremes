
[![Readthedocs Status](https://readthedocs.org/projects/xtremes/badge/?version=latest)](https://xtremes.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/xtremes.svg)](https://pypi.org/project/xtremes/)
![Python Versions](https://img.shields.io/pypi/pyversions/xtremes)
![License](https://img.shields.io/pypi/l/xtremes)
![Downloads](https://img.shields.io/pypi/dm/xtremes)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
[![Build Status](https://github.com/haufse/xtremes/actions/workflows/ci.yml/badge.svg)](https://github.com/haufse/xtremes/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/haufse/xtremes/branch/main/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/haufse/xtremes)


Welcome to xtremes!
===================

**xtremes** is a Python library designed for extreme value analysis, with tools for simulating time series, extracting block maxima, and performing advanced statistical operations, such as bootstrapping estimators for extreme value distributions. It was created as part of the ClimXtreme project and aims to provide supplementary code and simulations for related work.

Key Features:
-------------
- Simulates time series for extreme value distributions (GEV, Frechet, etc.).
- Extracts Disjoint and Sliding Block Maxima.
- Extracts Disjoint and Sliding Block High Order Statistics.
- Provides robust bootstrapping tools for extreme value statistics.
- Supports advanced MLE and PWM estimation for extreme value distributions.

Submodules:
-----------
- **HighOrderStatistics**: Contains functions and classes to compute block maxima, high-order statistics, and perform extreme value analysis.
- **bootstrap**: Provides methods for bootstrapping block maxima and sliding block maxima, with support for both Disjoint and Sliding Block Maxima methods.

Installation:
-------------
```bash
   (.venv) $ pip install xtremes
```

You can also view the package on PyPi at <https://pypi.org/project/xtremes/>.

Getting Started:
----------------
1. Install the package via pip:
   ```bash
   (.venv) $ pip install xtremes
   ```

2. Import the necessary submodules and start exploring extreme value statistics:
   ```python
   import xtremes.HighOrderStatistics as hos
   import xtremes.bootstrap as bst
   ```

3. For more detailed documentation, check out <https://xtremes.readthedocs.io/en/latest/>.

Example Usage:
--------------
Here's a simple example to get started with `xtremes`:

```python
import xtremes.HighOrderStatistics as hos
import xtremes.bootstrap as bst

# Simulate time series data
ts = hos.TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5])
ts.simulate(rep=10)

# Extract block maxima
ts.get_blockmaxima(block_size=5)

# Perform bootstrap analysis
bootstrap = bst.FullBootstrap(ts.values, block_size=5)
bootstrap.run_bootstrap(num_bootstraps=100)
print(bootstrap.statistics['mean'])
```

Documentation:
--------------
Further documentation can be found at <https://xtremes.readthedocs.io/en/latest/>.

Example Output:
---------------
The following plot shows block maxima extracted from a simulated time series:

![Block TopTwo Plot](images/MaxPicSBM.png)


Foundational insights behind the methods used in `xtremes.bootstrap` have been developed by [[BS24]].

Roadmap:
--------
- Add support for additional time series models (ARIMA).
- Improve documentation with more examples.
- Optimize bootstrapping methods for large datasets.

Note:
-----
This project is under active development throughout the project phase and will provide additional code to support theoretical advancements in extreme value statistics. (todo: add DOI (badge))

References:
-----------

[BS24]: Bücher, A., & Staud, T. (2024). Bootstrapping Estimators based on the Block Maxima Method. *arXiv preprint* [arXiv:2409.05529](https://arxiv.org/abs/2409.05529).


Suggested Citation:
-------------------
If you use the bootstrapping functionality or methods related to block maxima in this library, please cite the following paper:

```bibtex
@article{tba,  
  title={tba,  
  author={B{"u}cher, Axel and Haufs, Erik},  
  journal={tba},  
  year={tba}  
}

