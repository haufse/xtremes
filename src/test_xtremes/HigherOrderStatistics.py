import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import genextreme, invweibull, weibull_max, gumbel_r
from scipy.special import gamma as Gamma
from pynverse import inversefunc
import pickle
import test_xtremes.miscellaneous as misc

def extract_BM(timeseries, block_size=10, stride='DBM'):
    n = len(timeseries)
    r = block_size
    p = stride2int(stride, block_size)
    out = [np.max(timeseries[i*p:i*p+r]) for i in range((n-r)//p+1)]
    return np.array(out)

def extract_HOS(timeseries, orderstats=2, block_size=10, stride='DBM'):
    n = len(timeseries)
    r = block_size
    p = stride2int(stride, block_size)
    out = [np.sort(timeseries[i*p:i*p+r])[-orderstats:] for i in range((n-r)//p+1)]
    return np.array(out)

class TimeSeries:
    def __init__(self, n, distr='GEV', correlation='IID', modelparams=[0], ts=0):
        self.timeseries = []
        self.distr = distr
        self.corr = correlation
        self.modelparams = modelparams
        self.ts = ts
        self.len = n
        self.blockmaxima = []
        self.high_order_stats = []
    
    def __len__(self):
        return self.len
    
    def simulate(self, rep=10, seeds='default'):
        self.reps = 10
        if seeds == 'default':
            for i in range(rep):
                series = simulate_timeseries(self.len,
                                             distr=self.distr, 
                                             correlation=self.corr, 
                                             modelparams=self.modelparams, 
                                             ts=self.ts, 
                                             seed=i)
                self.timeseries.append(series)
        elif seeds == None:
            for i in range(rep):
                series = simulate_timeseries(self.len,
                                             distr=self.distr, 
                                             correlation=self.corr, 
                                             modelparams=self.modelparams, 
                                             ts=self.ts, 
                                             seed=None)
                self.timeseries.append(series)
        else:
            if not hasattr(seeds, '__len__'):
                # handles the case of seeds being an integer and rep=1
                seeds = [seeds]
            if len(seeds) != rep:
                raise ValueError('Number of given seeds does not match repititions!')
            else:
                for i in range(rep):
                    series = simulate_timeseries(self.len,
                                                distr=self.distr, 
                                                correlation=self.corr, 
                                                modelparams=self.modelparams, 
                                                ts=self.ts, 
                                                seed=seeds[i])
                    self.timeseries.append(series)
        
    def get_blockmaxima(self, block_size=2, stride='DBM', rep=10):
        if self.timeseries == []:
            warnings.warn('TimeSeries was not simulated yet, I will do it for you. For more flexibility, simulate first!')
            self.simulate(rep=rep)
        self.block_size = block_size
        self.stride = stride
        for series in self.timeseries:
            bms = extract_BM(series, block_size=block_size, stride=stride)
            self.blockmaxima.append(bms)
    

    def get_HOS(self, orderstats = 2, block_size=2, stride='DBM', rep=10):
        if self.timeseries == []:
            warnings.warn('TimeSeries was not simulated yet, I will do it for you. For more flexibility, simulate first!')
            self.simulate(rep=rep)
        self.block_size = block_size
        self.stride = stride
        for series in self.timeseries:
            hos = extract_HOS(series, orderstats=orderstats, block_size=block_size, stride=stride)
            self.high_order_stats.append(hos)

def cost_function(high_order_stats, option=1, estimate_pi=False, pi0=1):
    # define cost function based on option
    maxima = high_order_stats.T[-1]
    secondlargest = high_order_stats.T[-2]
    if option == 1:       
        def cost(params):
            gamma, mu, sigma, pi = params     
            uni, count = np.unique(maxima, return_counts=True)   
            cst = - np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option), count)
            return cst
        
    if option == 2:
        def cost(params):
            gamma, mu, sigma, pi = params     
            uni, count = np.unique(high_order_stats, return_counts=True, axis=0)   
            cst = - np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option), count)
            return cst
    
    if option == 3:
        def cost(params):
            gamma, mu, sigma, pi = params
            if not estimate_pi:
                pi = pi0
            # add likelihood for maxima
            uni, count = np.unique(maxima, return_counts=True)   
            cst = - np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option, max_only=True), count)
            # add likelihood for second largest
            uni, count = np.unique(secondlargest, return_counts=True)   
            cst -=  np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option, second_only=True), count)
            return cst
        
    return cost

def automatic_parameter_initialization(PWM_estimators, corr, ts=0.5):
    initParams = np.array(PWM_estimators)
    # pi parameter from theory
    if corr == 'ARMAX':
        pis = np.ones(len(PWM_estimators)) * misc.invsigmoid(1-ts)
    elif corr == 'IDD':
        pis = np.ones(len(PWM_estimators))
    else:
        pis = np.ones(shape=(1,len(PWM_estimators)))
    return np.append(initParams.T, pis, axis=0).T

class HighOrderStats:
    def __init__(self, TimeSeries):
        TimeSeries.get_HOS()
        self.TimeSeries = TimeSeries
        self.high_order_stats = TimeSeries.high_order_stats
        self.blockmaxima = [ho_stat.T[-1] for ho_stat in self.high_order_stats]
        self.gamma_true = modelparams2gamma_true(TimeSeries.distr, TimeSeries.corr, TimeSeries.modelparams)
        self.PWM_estimators = []
        self.ML_estimators = []
    
    def get_PWM_estimation(self):
        for bms in self.blockmaxima:
            b0, b1, b2 = misc.PWM_estimation(bms) 
            gamma, mu, sigma = misc.PWM2GEV(b0, b1, b2)
            self.PWM_estimators.append([gamma, mu, sigma])
        self.PWM_estimators = np.array(self.PWM_estimators)
        
    def get_ML_estimation(self, initParams = 'auto', option=1, estimate_pi=False):
        if self.PWM_estimators == []:
            self.get_PWM_estimation()
        if initParams == 'auto':
            initParams = automatic_parameter_initialization(self.PWM_estimators, 
                                                            self.TimeSeries.corr, 
                                                            ts=self.TimeSeries.ts)
        for i, ho_stat in enumerate(self.high_order_stats):
            cost = cost_function(ho_stat, option=option, estimate_pi=estimate_pi, pi0=pi0)
            results = misc.minimize(cost, initParams[i], method='Nelder-Mead')
    
            gamma, mu, sigma, pi = results.x
            
            self.ML_estimators.append([gamma, mu, sigma, pi])
        self.ML_estimators = np.array(self.ML_estimators)
        

# time series parameter
ts = 0.5
#gamma_true = 1

# 0 as lat parameter: no Time Series Dependence

def execute_model(model='GPD', ts=0, gamma_true=0, estimate_pi=False, option=1):
    ns = [1000, 2000, 3000]
    all_mse, all_gamma, all_mu, all_sigma, all_pi = [], [], [], [], []
    for n in ns:
        n_mse, n_gamma, n_mu, n_sigma, n_pi = [], [], [], [], []
        rs=[10, 20, 30, 40, 50, 60, 70, 80]
        for r in tqdm(rs):
            mses, gammas, mus, sigmas, pis = [], [], [], [], []
            for p in [r, 1]:
                gamma, mu, sigma, pi = getgammas(n=n, r=r, rep=200, model=model, 
                                            modelparams=modelparams_dict[model], 
                                            stride=p, initParams='auto', estimate_pi=estimate_pi, option=option)
                MSE_sl, variance_sl, bias_sl = misc.mse(gamma, gamma_true)
                
                mses.append(MSE_sl)
                gammas.append(gamma)
                mus.append(mu)
                sigmas.append(sigma)
                #pis.append(sigmoid(pi))
                pis.append(pi)
            n_mse.append(mses)
            n_gamma.append(gammas)
            n_mu.append(mus)
            n_sigma.append(sigmas)
            n_pi.append(pis)
        all_mse.append(n_mse)
        all_gamma.append(n_gamma)
        all_mu.append(n_mu)
        all_sigma.append(n_sigma)
        all_pi.append(n_pi)

    summary_dict = {
        'mse': all_mse,
        'gamma': all_gamma,
        'mu': all_mu,
        'sigma': all_sigma,
        'pi': all_pi
    }

    with open(model+str(int(10*gamma_true))+'.pkl', 'wb') as f:
        pickle.dump(summary_dict, f)
