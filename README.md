
[![Readthedocs Status](https://readthedocs.org/projects/xtremes/badge/?version=latest)](https://xtremes.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/xtremes.svg)](https://pypi.org/project/xtremes/)
![Python Versions](https://img.shields.io/pypi/pyversions/xtremes)
![License](https://img.shields.io/pypi/l/xtremes)
![Downloads](https://img.shields.io/pypi/dm/xtremes)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
[![Build Status](https://github.com/haufse/xtremes/actions/workflows/ci.yml/badge.svg)](https://github.com/haufse/xtremes/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/haufse/xtremes/branch/main/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/haufse/xtremes)
[![arXiv](https://img.shields.io/badge/arXiv-2502.15036-b31b1b.svg)](https://arxiv.org/abs/2502.15036)


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
- **topt**: Contains functions and classes to compute block maxima, high-order statistics, and perform extreme value analysis.
- **biascorrection**: Implements tools for bias-correcting Top-$t$ Pseudo-MLEs as described in [[BH25]]
- **miscellaneous**: Provides supplementary functions for other modules
- **bootstrap**: Provides methods for bootstrapping block maxima and sliding block maxima, with support for both Disjoint and Sliding Block Maxima methods, developed by [[BS24]].

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
   
   import xtremes as xx
   import xtremes.topt as topt
   ...
   ```

3. For more detailed documentation, check out <https://xtremes.readthedocs.io/en/latest/>.

Example Usage:
--------------
Here's a simple example to get started with `xtremes`:

```python
import xtremes.topt as topt

# Simulate time series data
k, bs = 100, 100
ts = topt.TimeSeries(n=k*bs, distr='Pareto', correlation='IID', modelparams=[0.5])
ts.simulate(rep=10)

# Extract block maxima
ts.get_blockmaxima(block_size=bs)
# Extract Sliding Top-Two
ts.get_HOS(orderstats=2, block_size=bs, stride='SBM')

# initialize the HighOrderStats class
hos = topt.HighOrderStats(ts)
# perform Maximum Likelihood estimation
HOS.get_ML_estimation(r=2, FrechetOrGEV='Frechet')
print(HOS.ML_estimators.values)

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
- Implement biascorrection for $t \geq 3$
- Implement tools to choose number of high order statistics data-adaptively
- Other projects yet to come! 

Note:
-----
This project is under active development throughout the project phase and will provide additional code to support theoretical advancements in extreme value statistics. The submodules will be sorted to the papers yet to come.

References:
-----------

- [BS24]: Bücher, A., & Staud, T. (2024). Bootstrapping Estimators based on the Block Maxima Method. *arXiv preprint* [arXiv:2409.05529](https://arxiv.org/abs/2409.05529),
- [BH25]: Bücher, A., & Haufs, E. (2025). Extreme Value Analysis based on Blockwise Top-Two Order Statistics. *arXiv preprint* [arXiv:2502.15036](https://arxiv.org/abs/2502.15036).


Suggested Citation:
-------------------
If you use the functionalities related to fitting a MLE to blockwise high order statistics, please cite the following paper:

```bibtex
@misc{bücherhaufs2025toptwo,
      title={Extreme Value Analysis based on Blockwise Top-Two Order Statistics}, 
      author={Axel Bücher and Erik Haufs},
      year={2025},
      eprint={2502.15036},
      archivePrefix={arXiv},
      primaryClass={math.ST},
      url={https://arxiv.org/abs/2502.15036}, 
}

