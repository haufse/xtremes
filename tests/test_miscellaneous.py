import numpy as np
import pytest
from xtremes.miscellaneous import (sigmoid, invsigmoid, mse, GEV_cdf, GEV_pdf, GEV_ll,
                           PWM_estimation, PWM2GEV, simulate_timeseries, 
                           stride2int, modelparams2gamma_true)

# Test sigmoid function
def test_sigmoid_basic():
    x = np.array([-2, -1, 0, 1, 2])
    result = sigmoid(x)
    expected = np.array([0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708])
    assert np.allclose(result, expected, atol=1e-7), "Sigmoid function output incorrect"

# Test invsigmoid function
def test_invsigmoid_basic():
    y = np.array([0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708])
    result = invsigmoid(y)
    expected = np.array([-2, -1, 0, 1, 2])
    assert np.allclose(result, expected, atol=1e-7), "Inverse sigmoid function output incorrect"

def test_invsigmoid_invalid():
    with pytest.raises(ValueError):
        invsigmoid(-0.1)

# Test mse function
def test_mse_basic():
    gammas = [0.5, 0.6, 0.7]
    gamma_true = 0.6
    mse_val, variance, bias = mse(gammas, gamma_true)
    assert mse_val > 0, "MSE should be positive"
    assert variance > 0, "Variance should be positive"
    assert bias >= 0, "Bias should be non-negative"

def test_mse_single_value():
    gammas = [0.5]
    mse_val, variance, bias = mse(gammas, 0.6)
    assert np.isnan(mse_val), "MSE should be NaN for single value input"

# Test GEV_cdf function
def test_GEV_cdf_basic():
    x = np.array([1, 2, 3])
    result = GEV_cdf(x, gamma=0.5, mu=2, sigma=1, theta=0.8)
    # expected = np.array([0.54610814, 0.62171922, 0.69703039])
    assert np.all(result >=0) & np.all(result <=1), "GEV CDF output incorrect"

# Test GEV_pdf function
def test_GEV_pdf_basic():
    x = np.array([1, 2, 3])
    result = GEV_pdf(x, gamma=0.5, mu=2, sigma=1)
    # expected = np.array([0.17603266, 0.23254416, 0.28556358])
    assert np.all(result >=0) & np.all(result <=1), "GEV PDF output incorrect"

# Test GEV_ll function
def test_GEV_ll_basic():
    x = np.array([1, 2, 3])
    result = GEV_ll(x, gamma=0.5, mu=2, sigma=1)
    # expected = np.array([-3.11578562, -2.23851549, -1.85551157])
    assert np.all(result <=0), "GEV log-likelihood output incorrect"

# Test PWM_estimation function
def test_PWM_estimation_basic():
    maxima = np.array([5, 8, 12, 15, 18])
    result = PWM_estimation(maxima)
    # expected = (11.6, 11.2, 39.2)
    assert len(result)==3, "PWM estimation output incorrect"

# Test PWM2GEV function
def test_PWM2GEV_basic():
    b_0, b_1, b_2 = 10, 20, 30
    gamma, mu, sigma = PWM2GEV(b_0, b_1, b_2)
    assert isinstance(gamma, float), "PWM to GEV should return a float for gamma"
    assert isinstance(mu, float), "PWM to GEV should return a float for mu"
    assert isinstance(sigma, float), "PWM to GEV should return a float for sigma"

# Test simulate_timeseries function
def test_simulate_timeseries_basic():
    ts = simulate_timeseries(100, distr='GEV', correlation='IID', modelparams=[0])
    assert len(ts) == 100, "Simulated time series should have length 100"

def test_simulate_timeseries_invalid_correlation():
    with pytest.raises(ValueError):
        simulate_timeseries(100, correlation='invalid')

# Test stride2int function
def test_stride2int_basic():
    assert stride2int('SBM', 10) == 1, "Stride SBM should return 1"
    assert stride2int('DBM', 10) == 10, "Stride DBM should return block size"
    assert stride2int(5, 10) == 5, "Integer stride should return the same integer"

def test_stride2int_invalid():
    with pytest.raises(TypeError):
        stride2int('invalid', 10)

# Test modelparams2gamma_true function
def test_modelparams2gamma_true_gev():
    gamma_true = modelparams2gamma_true('GEV', 'IID', [0.5])
    assert gamma_true == 0.5, "GEV gamma_true extraction incorrect"

def test_modelparams2gamma_true_invalid():
    with pytest.raises(ValueError):
        modelparams2gamma_true('Normal', 'IID', [0.5])

def test_all_functions():
    test_sigmoid_basic()
    test_invsigmoid_basic()
    test_invsigmoid_invalid()
    test_mse_basic()
    test_mse_single_value()
    test_GEV_cdf_basic()
    test_GEV_pdf_basic()
    test_GEV_ll_basic()
    test_PWM_estimation_basic()
    test_PWM2GEV_basic()
    test_simulate_timeseries_basic()
    test_simulate_timeseries_invalid_correlation()
    test_stride2int_basic()
    test_stride2int_invalid()
    test_modelparams2gamma_true_gev()
    test_modelparams2gamma_true_invalid()

    print("All tests passed!")