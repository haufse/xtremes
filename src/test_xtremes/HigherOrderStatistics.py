import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import genextreme, invweibull, weibull_max, gumbel_r
from scipy.special import gamma as Gamma
from pynverse import inversefunc
import pickle

import test_xtremes.miscellaneous as misc

def getgammas(n=200, r=10, rep=1000, model='normal', modelparams = [0,1, 0], stride='block', initParams='auto', estimate_pi=False, option=1):
    """
    params:
        n: number of data points
        r: block size
        rep: repititions for estimation
        model:  model to be chosen
            modelparams: list of necessary parameters
            'normal': iid normal, param1 = mean, param2 = std
            'exponential': iid exponential, param1 = lambda
            'AR':
            'AMAX':
        stride: sliding block with stride
            'block': stride = r, i.e. disjoint blocks
            1: sliding blocks
            other ints: sliding with stride
        initParams: initial params for ML optimization
    """
    gammas, mus, sigmas, pis = [], [], [], []
    for i in range(rep):# tqdm
        succ = False
        attempt_counter = 0
        while succ == False:
            
            s = misc.draw_randoms(modelname=model, modelparams=modelparams, n=n)

            if stride == 'block':
                p = r
            elif isinstance(stride, int):
                p = stride
            else:
                raise ValueError('No valid stride specified')
            
            if initParams == 'auto':
                mxm = [np.max(s[i*p:i*p+r]) for i in range((n-r)//p)]
                b_0, b_1, b_2 = misc.PWM_estimation(mxm)
                gamma0, mu0, sigma0 = misc.PWM2GEV(b_0, b_1, b_2)
                # last model param is TS param, Ex 10.12 in Beirlant et al 2004 states pi = ts in AMAX
                pi0 = misc.invsigmoid(modelparams[-1]) #np.random.uniform(0.7, 1)
                initParams = [gamma0, mu0, sigma0, pi0]

            # define cost function based on option
            if option == 1:
                
                maxima = [np.max(s[i*p:i*p+r]) for i in range((n-r)//p)]
                
                #def cost(params):
                #    gamma, mu, sigma, pi = params        
                #    cst = - sum(ll_GEV(maxima, gamma=gamma,mu=mu,sigma=sigma, pi=pi))
                #    return cst
                
                def cost(params):
                    gamma, mu, sigma, pi = params     
                    uni, count = np.unique(maxima, return_counts=True)   
                    cst = - np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option), count)
                    return cst
            
            if option == 2:
                
                maxima = np.array([[np.max(s[i*p:i*p+r]), np.partition(s[i*p:i*p+r], -2)[-2]] for i in range((n-r)//p)])
                
                def cost(params):
                    gamma, mu, sigma, pi = params     
                    uni, count = np.unique(maxima, return_counts=True,axis=0)   
                    cst = - np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option), count)
                    return cst
            
            if option == 3:
                
                maxima_second = np.array([[np.max(s[i*p:i*p+r]), np.partition(s[i*p:i*p+r], -2)[-2]] for i in range((n-r)//p)])


                #def cost(params):
                #    gamma, mu, sigma, pi = params     
                #    uni, count = np.unique(maxima_second, return_counts=True,axis=0)   
                #    cst = - np.dot(ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option), count)
                #    return cst
                
                maxima, second = maxima_second.T
                def cost(params):
                    gamma, mu, sigma, pi = params
                    if not estimate_pi:
                        pi = pi0
                    # add likelihood for maxima
                    uni, count = np.unique(maxima, return_counts=True)   
                    cst = - np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option, max_only=True), count)
                    # add likelihood for second maxima
                    uni, count = np.unique(second, return_counts=True)   
                    cst -=  np.dot(misc.ll_GEV(uni, gamma=gamma,mu=mu,sigma=sigma, pi=pi, option=option, second_only=True), count)
                    return cst


            results = minimize(cost, initParams, method='Nelder-Mead')#'BFGS')#options={'maxiter':10000})
            succ = results.success
            attempt_counter += 1
            if attempt_counter > 5:
                print('Warning: Minimization failed more than 5 times, breaking...')
                break
        if attempt_counter < 6:
            gamma, mu, sigma, pi = results.x
            gammas.append(gamma)
            mus.append(mu)
            sigmas.append(sigma)
            pis.append(pi)
    return gammas, mus, sigmas, pis


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
