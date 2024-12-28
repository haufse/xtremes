import numpy as np
import pytest
from xtremes.bootstrap import circmax, uniquening, Bootstrap, aggregate_boot, ML_Estimator, FullBootstrap

# Test circmax function
def test_circmax_dbm():
    sample = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = circmax(sample, bs=5, stride='DBM')
    assert np.array_equal(result, np.array([5, 10])), "DBM block maxima extraction failed"

def test_circmax_sbm():
    sample = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = circmax(sample, bs=3, stride='SBM')
    expected = np.array([[3, 4, 5, 6, 6, 6]])
    assert np.array_equal(result, expected), "SBM block maxima extraction failed"

def test_circmax_invalid_stride():
    sample = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        circmax(sample, bs=2, stride='invalid')


# Test uniquening function
def test_uniquening():
    sample = np.array([[1, 2, 2, 3], [1, 1, 2, 2]])
    result = uniquening(sample)
    assert len(result) == 2, "There should be 2 arrays of unique values"
    assert np.array_equal(result[0][0], np.array([1, 2, 3])), "First array unique values incorrect"
    assert np.array_equal(result[0][1], np.array([1, 2, 1])), "First array counts incorrect"


# Test Bootstrap function
def test_bootstrap():
    sample = [1, 2, 3, 4, 5]
    result = Bootstrap(sample)
    assert len(result) == 5, "Bootstrap sample should have the same length as the original"
    assert set(result).issubset(set(sample)), "Bootstrap sample should only contain elements from the original"


# Test aggregate_boot function
def test_aggregate_boot():
    boot_samp = [(np.array([1, 2, 3]), np.array([1, 1, 2])), (np.array([2, 3]), np.array([2, 1]))]
    result = aggregate_boot(boot_samp)
    expected = np.array([[1, 1], [2, 3], [3, 3]])
    assert np.array_equal(result, expected), "Aggregate boot result is incorrect"


# Test ML_Estimator class
def test_ML_Estimator_frechet():
    data = np.array([[5, 2], [10, 3], [15, 4]])
    estimator = ML_Estimator(data)
    params = estimator.maximize_frechet()
    assert len(params) == 2, "Frechet estimator should return two parameters"
    assert params[0] > 0 and params[1] > 0, "Frechet parameters should be positive"

def test_ML_Estimator_gev():
    data = np.array([[5, 2], [10, 3], [15, 4]])
    estimator = ML_Estimator(data)
    params = estimator.maximize_gev()
    assert len(params) == 3, "GEV estimator should return three parameters"
    assert params[1] > 0, "GEV scale parameter should be positive"


# Test FullBootstrap class
def test_FullBootstrap_frechet():
    np.random.seed(0)
    sample = np.random.rand(100)
    bootstrap = FullBootstrap(sample, bs=10, stride='DBM', dist_type='Frechet')
    bootstrap.run_bootstrap(num_bootstraps=10)
    assert 'mean' in bootstrap.statistics, "Bootstrap statistics should include 'mean'"
    assert 'std' in bootstrap.statistics, "Bootstrap statistics should include 'std'"
    assert len(bootstrap.values) == 10, "There should be 10 bootstrap estimates"

def test_FullBootstrap_gev():
    np.random.seed(1)
    sample = np.random.rand(100)
    bootstrap = FullBootstrap(sample, bs=10, stride='DBM', dist_type='GEV')
    bootstrap.run_bootstrap(num_bootstraps=10)
    assert 'mean' in bootstrap.statistics, "Bootstrap statistics should include 'mean'"
    assert 'std' in bootstrap.statistics, "Bootstrap statistics should include 'std'"
    assert len(bootstrap.values) == 10, "There should be 10 bootstrap estimates"

if __name__ == "__main__":
    test_circmax_dbm()
    test_circmax_sbm()
    test_circmax_invalid_stride()
    test_uniquening()
    test_bootstrap()
    test_aggregate_boot()
    test_ML_Estimator_frechet()
    test_ML_Estimator_gev()
    test_FullBootstrap_frechet()
    test_FullBootstrap_gev()
    
    print("All tests passed!")