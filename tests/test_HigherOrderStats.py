import numpy as np
import pytest
from xtremes.HigherOrderStatistics import log_likelihood, Frechet_log_likelihood, extract_BM, extract_HOS, automatic_parameter_initialization, run_ML_estimation

# Test log_likelihood function
def test_log_likelihood_basic():
    high_order_statistics = np.array([[0.1, 0.2], [0.3, 0.4]])
    ll_value = log_likelihood(high_order_statistics, gamma=0.5, mu=0, sigma=1, r=2)
    assert isinstance(ll_value, float), "Log likelihood should return a float value"

def test_log_likelihood_invalid_sigma():
    high_order_statistics = np.array([[0.1, 0.2], [0.3, 0.4]])
    ll_value = log_likelihood(high_order_statistics, gamma=0.5, mu=0, sigma=-1, r=2)
    assert ll_value != float('inf'), "Sigma should be handled to avoid infinite values"

def test_log_likelihood_default_parameters():
    high_order_statistics = np.array([[0.1, 0.2], [0.3, 0.4]])
    ll_value = log_likelihood(high_order_statistics)
    assert isinstance(ll_value, float), "Log likelihood should work with default parameters"

# Test Frechet_log_likelihood function
def test_Frechet_log_likelihood_basic():
    high_order_statistics = np.array([[0.5, 1.0], [1.5, 2.0]])
    ll_value = Frechet_log_likelihood(high_order_statistics, alpha=2, sigma=1.5, r=2)
    assert isinstance(ll_value, float), "Frechet log likelihood should return a float value"

def test_Frechet_log_likelihood_invalid_alpha():
    high_order_statistics = np.array([[0.5, 1.0], [1.5, 2.0]])
    ll_value = Frechet_log_likelihood(high_order_statistics, alpha=0, sigma=1.5, r=2)
    assert ll_value == float('-inf'), "Alpha less than or close to 0 should result in negative infinity"

# Test extract_BM function
def test_extract_BM_basic():
    timeseries = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = extract_BM(timeseries, block_size=2, stride='DBM')
    assert np.array_equal(result, np.array([2, 4, 6, 8, 10])), "Block maxima extraction is incorrect"

def test_extract_BM_with_indices():
    timeseries = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    indices, result = extract_BM(timeseries, block_size=2, stride='DBM', return_indices=True)
    assert np.array_equal(result, np.array([2, 4, 6, 8, 10])), "Block maxima extraction with indices is incorrect"
    assert np.array_equal(indices, np.array([1, 3, 5, 7, 9])), "Indices of maxima are incorrect"

# Test extract_HOS function
def test_extract_HOS_basic():
    timeseries = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = extract_HOS(timeseries, orderstats=2, block_size=5, stride='DBM')
    expected = np.array([[4, 5], [9, 10]])
    assert np.array_equal(result, expected), "High order statistics extraction is incorrect"

# Test automatic_parameter_initialization
def test_automatic_parameter_initialization():
    PWM_estimators = np.array([0.1, 0.2, 0.3])
    params = automatic_parameter_initialization(PWM_estimators, corr='ARMAX', ts=0.8)
    assert isinstance(params, np.ndarray), "Parameter initialization should return a numpy array"
    assert np.array_equal(params, PWM_estimators), "Automatic initialization should return input PWM estimators"

# Test run_ML_estimation (note: mock the file reading part if needed)
@pytest.mark.skip(reason="Depends on file input, needs mocking or appropriate test data")
def test_run_ML_estimation():
    result = run_ML_estimation("timeseries_data.pkl", corr='IID', gamma_true=0.5, block_sizes=[10, 20, 30])
    assert isinstance(result, dict), "The result should be a dictionary"

# classes
from xtremes.HigherOrderStatistics import TimeSeries, PWM_estimators, ML_estimators, HighOrderStats, Data


# Test TimeSeries class
def test_time_series_initialization():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    assert ts.len == 100, "TimeSeries length should be 100"
    assert ts.distr == 'GEV', "Distribution should be 'GEV'"
    assert ts.corr == 'IID', "Correlation should be 'IID'"
    assert ts.modelparams == [0.5], "Model params should be [0.5]"
    assert ts.ts == 0.6, "ts parameter should be 0.6"


def test_time_series_simulate():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=5, seeds=[42, 43, 44, 45, 46])
    assert len(ts.values) == 5, "There should be 5 simulations"
    assert len(ts.values[0]) == 100, "Each time series should have 100 values"
    

def test_time_series_get_blockmaxima():
    ts = TimeSeries(n=10, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=1, seeds=[42])
    ts.get_blockmaxima(block_size=2, stride='DBM')
    assert len(ts.blockmaxima) == 1, "There should be 1 set of block maxima"
    assert len(ts.blockmaxima[0]) == 5, "Block maxima should be of size 5 for block_size=2"

    
def test_time_series_get_HOS():
    ts = TimeSeries(n=10, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=1, seeds=[42])
    ts.get_HOS(orderstats=2, block_size=5, stride='DBM')
    assert len(ts.high_order_stats) == 1, "There should be 1 set of high order statistics"
    assert ts.high_order_stats[0].shape == (2, 2), "HOS should be of size (2,2) for block_size=5 and orderstats=2"


# Test PWM_estimators class
def test_PWM_estimators_initialization():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=1, seeds=[42])
    ts.get_blockmaxima(block_size=2, stride='DBM')
    pwm = PWM_estimators(ts)
    assert pwm.blockmaxima is not None, "Block maxima should be assigned"
    assert len(pwm.blockmaxima) > 0, "Block maxima should not be empty"


def test_PWM_estimators_get_PWM_estimation():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=1, seeds=[42])
    ts.get_blockmaxima(block_size=2, stride='DBM')
    pwm = PWM_estimators(ts)
    pwm.get_PWM_estimation()
    assert pwm.values is not None, "PWM estimators should be computed"
    assert len(pwm.values) > 0, "PWM estimators should not be empty"


def test_PWM_estimators_get_statistics():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=3, seeds=[42,1,2])
    ts.get_blockmaxima(block_size=2, stride='DBM')
    pwm = PWM_estimators(ts)
    pwm.get_PWM_estimation()
    pwm.get_statistics(gamma_true=0.1)
    assert 'gamma_mean' in pwm.statistics, "Statistics should include 'gamma_mean'"
    assert isinstance(pwm.statistics['gamma_mean'], float), "'gamma_mean' should be a float"


# Test ML_estimators class
def test_ML_estimators_initialization():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=1, seeds=[42])
    ts.get_HOS(orderstats=2, block_size=5, stride='DBM')
    ml = ML_estimators(ts)
    assert ml.high_order_stats is not None, "High order stats should be assigned"
    assert len(ml.high_order_stats) > 0, "High order stats should not be empty"


def test_ML_estimators_get_ML_estimation():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=1, seeds=[42])
    ts.get_HOS(orderstats=2, block_size=5, stride='DBM')
    pwm = PWM_estimators(ts)
    pwm.get_PWM_estimation()
    ml = ML_estimators(ts)
    ml.get_ML_estimation(PWM_estimators=pwm)
    assert ml.values is not None, "ML estimators should be computed"
    assert len(ml.values) > 0, "ML estimators should not be empty"


# Test HighOrderStats class
def test_HighOrderStats_initialization():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=1, seeds=[42])
    ts.get_HOS(orderstats=2, block_size=5, stride='DBM')
    hos = HighOrderStats(ts)
    assert hos.high_order_stats is not None, "High order stats should be assigned"
    assert len(hos.high_order_stats) > 0, "High order stats should not be empty"


def test_HighOrderStats_get_PWM_estimation():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=1, seeds=[42])
    ts.get_HOS(orderstats=2, block_size=5, stride='DBM')
    hos = HighOrderStats(ts)
    hos.get_PWM_estimation()
    assert hos.PWM_estimators.values is not None, "PWM estimators should be computed in HighOrderStats"
    assert len(hos.PWM_estimators.values) > 0, "PWM estimators should not be empty in HighOrderStats"


def test_HighOrderStats_get_ML_estimation():
    ts = TimeSeries(n=100, distr='GEV', correlation='IID', modelparams=[0.5], ts=0.6)
    ts.simulate(rep=1, seeds=[42])
    ts.get_HOS(orderstats=2, block_size=5, stride='DBM')
    hos = HighOrderStats(ts)
    hos.get_PWM_estimation()
    hos.get_ML_estimation(FrechetOrGEV='GEV')
    assert hos.ML_estimators.values is not None, "ML estimators should be computed in HighOrderStats"
    assert len(hos.ML_estimators.values) > 0, "ML estimators should not be empty in HighOrderStats"


# Test Data class
def test_Data_initialization():
    data = np.random.normal(loc=10, scale=2, size=100)
    data_obj = Data(data)
    assert len(data_obj) == 100, "Data length should be 100"
    assert data_obj.values is not None, "Values should be assigned"


def test_Data_get_blockmaxima():
    data = np.random.normal(loc=10, scale=2, size=100)
    data_obj = Data(data)
    data_obj.get_blockmaxima(block_size=10, stride='DBM')
    assert len(data_obj.blockmaxima) > 0, "Block maxima should be computed"
    assert len(data_obj.bm_indices) == len(data_obj.blockmaxima), "Indices and blockmaxima should be of equal length"


def test_Data_get_HOS():
    data = np.random.normal(loc=10, scale=2, size=100)
    data_obj = Data(data)
    data_obj.get_HOS(orderstats=2, block_size=10, stride='DBM')
    assert len(data_obj.high_order_stats) > 0, "High order statistics should be computed"
    assert data_obj.high_order_stats.shape == (10, 2), "HOS shape should be correct for block size 10"


def test_Data_get_ML_estimation():
    data = np.random.normal(loc=10, scale=2, size=100)
    data_obj = Data(data)
    data_obj.get_HOS(orderstats=1, block_size=10, stride='DBM')
    data_obj.get_ML_estimation(FrechetOrGEV='GEV')
    assert data_obj.ML_estimators.values is not None, "ML estimators should be computed in Data class"
    assert len(data_obj.ML_estimators.values) > 0, "ML estimators should not be empty in Data class"

if __name__ == "__main__":
    test_log_likelihood_basic()
    test_log_likelihood_invalid_sigma()
    test_log_likelihood_default_parameters()
    test_Frechet_log_likelihood_basic()
    test_Frechet_log_likelihood_invalid_alpha()
    test_extract_BM_basic()
    test_extract_BM_with_indices()
    test_extract_HOS_basic()
    test_automatic_parameter_initialization()
    test_run_ML_estimation()
    test_time_series_initialization()
    test_time_series_simulate()
    test_time_series_get_blockmaxima()
    test_time_series_get_HOS()
    test_PWM_estimators_initialization()
    test_PWM_estimators_get_PWM_estimation()
    test_PWM_estimators_get_statistics()
    test_ML_estimators_initialization()
    test_ML_estimators_get_ML_estimation()
    test_HighOrderStats_initialization()
    test_HighOrderStats_get_PWM_estimation()
    test_HighOrderStats_get_ML_estimation()
    test_Data_initialization()
    test_Data_get_blockmaxima()
    test_Data_get_HOS()
    test_Data_get_ML_estimation()

    print("All tests passed!")