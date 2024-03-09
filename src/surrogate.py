import smt
import chaospy as cp

import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS
from smt.surrogate_models import KRG

from scipy.optimize import minimize
from scipy import stats

from tqdm import tqdm

rng = np.random.default_rng()

def get_surrogate(y, sampling, N):
    xt = sampling(N)
    
    GP = KRG(theta0=[1e-2]*np.ones(sampling.options['xlimits'].shape[0]), print_prediction=False, print_global=False)

    GP.set_training_values(xt, y(xt[:, 0], xt[:, 1]))
    GP.train()

    return xt, GP

"""
    Sample from the surrogare.

    @param surrogate A KRG GP.
    @param xi Samples from standard normal distribution. Numpy array with shape (N,)
    @param z  Z value of points. Either float or numpy array of size (N,)
    @param x  X values of points. Numpy array of size (N,)
    
"""
def sample_surrogate_uniform(surrogate, xi, z, x):
    if type(z) is float or z.size == 1:
        z = z*np.ones_like(x)
    mean = surrogate.predict_values(np.array([z, x]).T)[0, 0]
    var = surrogate.predict_variances(np.array([z, x]).T)[0, 0]
    return mean + xi*np.sqrt(var)

def solve_surrogate(surrogates, xi, z):
    # solution, assume length 2 for now
    y = np.zeros((xi.shape[0], 2))
    prev_y = y.copy()
    for i in range(1000): # until convergence
        prev_y[:] = y
        y[:, 0] = sample_surrogate_uniform(surrogates[0], xi[:, 0], z, y[:, 1])
        y[:, 1] = sample_surrogate_uniform(surrogates[1], xi[:, 1], z, y[:, 0])
    print('Convergence %', np.sum(np.abs(prev_y-y)<1e-6)/(2*y.shape[0]))
    print(y.shape)
    return y

def sample_surrogate_solution(surrogates, z, N=1):
    res = []
    xi = rng.normal(size=(N, 2))
    print(xi.shape, xi[:10, :])
    res.append(solve_surrogate(surrogates, xi, z))
    res = np.array(res).squeeze()

    return res

def fit_PCE(surrogates, f_obj, z, order, N=100):
    gaussian = cp.J(cp.Normal(0., 1.), cp.Normal(0., 1.))

    samples = gaussian.sample(N)
    f_val = []

    f_val = f_obj(z, *solve_surrogate(surrogates, samples.T, z).T)
    
    expansion = cp.generate_expansion(order, gaussian)
    model, coeff_hermite = cp.fit_regression(expansion, samples, f_val, retall=1)

    return {'model': model, 'coeff_hermite': coeff_hermite, 'gaussian': gaussian, 'expansion': expansion}

def construct_KL_GP(surrogates, f_obj, z_DOE, order, N=100):
    models_z = []
    A = []
    gaussian = None
    expansion = None
    
    for z in z_DOE:
        PCE = fit_PCE(surrogates, f_obj, z, order, N)
        model, coeff_hermite, gaussian, expansion = PCE['model'], PCE['coeff_hermite'], PCE['gaussian'], PCE['expansion']
        
        models_z.append(model)
        A.append(coeff_hermite)

    A = np.array(A)
    K = np.zeros((len(z_DOE), len(z_DOE)))

    for i in range(1, A.shape[1]):
        K += A[:, i:i+1] @ A[:, i:i+1].T

    eigval, phi = np.linalg.eig(K)

    # interpolate mean and eigenvectors, according to Dubreuil et al. (2020)
    phi_GP = [KRG(theta0=[1e-2], print_prediction=False, print_global=False) for i in range(len(z_DOE)+1)]

    phi_GP[0].set_training_values(z_DOE, A[:, 0])
    phi_GP[0].train()

    for coeff_i, phi_i in enumerate(phi_GP[1:]):
        phi_i.set_training_values(z_DOE, phi[coeff_i])
        phi_i.train()
    return {'A': A, 'phi': phi, 'phi_GP': phi_GP, 'gaussian': gaussian, 'expansion': expansion}

"""
    Returns the PCE expansions at the points z_i that were part of the DOE
    
"""
def PCE_z_DOE(KL_GP):
    A, phi, expansion = KL_GP['A'], KL_GP['phi'], KL_GP['expansion']

    PCEs = []
    for z_i in range(len(phi)):
        PCE = KL_GP['A'][z_i, 0]*KL_GP['expansion'][0]
    
        for k in range(len(phi)):
            for j in range(1, A.shape[1]):
                PCE = PCE + (A[:, j] @ phi[k])*expansion[j]*phi[k, z_i]
        PCEs.append(PCE)
    return PCEs
"""
    Returns the PCE expansion at an arbitrary point z 
"""
def PCE_z(KL_GP, z):
    A, phi, phi_GP, gaussian, expansion = KL_GP['A'], KL_GP['phi'], KL_GP['phi_GP'], KL_GP['gaussian'], KL_GP['expansion']
    
    eta = gaussian.sample(1)
    # is this the correct way of sampling from the random field?
    PCE = (phi_GP[0].predict_values(np.array([z])) + eta[0]*phi_GP[0].predict_variances(np.array([z]))).item() * expansion[0]
    for k in range(len(phi)):
        eta = gaussian.sample(1) 
        for j in range(1, A.shape[1]):
            PCE = PCE + (A[:, j] @ phi[k])*(phi_GP[k+1].predict_values(np.array([z])) + eta[0]*phi_GP[k+1].predict_variances(np.array([z]))).item()*expansion[j]
           
    return PCE

"""
    Returns the PCE expansion at an arbitrary point z 
"""
def PCE_calc(KL_GP, z, xi):
    A, phi, phi_GP, gaussian, expansion = KL_GP['A'], KL_GP['phi'], KL_GP['phi_GP'], KL_GP['gaussian'], KL_GP['expansion']
    
    eta = gaussian.sample(xi.shape[1])
    # is this the correct way of sampling from the random field?
    vals = (phi_GP[0].predict_values(np.array([z])) + eta[0]*phi_GP[0].predict_variances(np.array([z]))) * expansion[0](*xi)
    for k in range(len(phi)):
        eta = gaussian.sample(xi.shape[1]) 
        for j in range(1, A.shape[1]):
            vals += (A[:, j] @ phi[k])*(phi_GP[k+1].predict_values(np.array([z])) + eta[0]*phi_GP[k+1].predict_variances(np.array([z])))*expansion[j](*xi)
           
    return vals

def EI(KL_GP, z, PCE_DOE=None, *args):

    if PCE_DOE is None:
        PCE_DOE = PCE_z_DOE(KL_GP)
    
    samples = KL_GP['gaussian'].sample(10000)
    
    # verify that PCE is the same as before 
    #print(PCE, samples.shape, *samples)
    y_z = PCE_calc(KL_GP, z, samples)

    y_DOE = PCE_DOE[0](*samples)

    for pce_ in PCE_DOE[1:]:
        y_DOE = np.minimum(y_DOE, pce_(*samples))

    y = y_DOE - y_z
    y[np.where(y<0, True, False)] = 0
    return np.mean(y)
    mean_z = np.mean(y_z)
    std_z = np.std(y_z, ddof=1)
    return np.sum((y_DOE - mean_z) * stats.norm.cdf((y_DOE - mean_z)/std_z) + std_z*stats.norm.pdf((y_DOE - mean_z)/std_z))/samples.shape[0]

def find_z_next(KL_GP):
    z_range = [-5, 5]
    def EI_(z, *args):
        # n inputs
        z = z_range[0] + ((z_range[1]-z_range[0])*z)
        return -EI(KL_GP, z)
    z_min = minimize(EI_, rng.uniform(0., 1.), method='COBYLA', options={'rhobeg': 0.5}, bounds=[(0., 1.)])
    print(z_range[0] + ((z_range[1]-z_range[0])*z_min.x), z_min.fun)
    for i in tqdm(range(19)):
        z_new = minimize(EI_, rng.uniform(0., 1.), method='COBYLA', options={'rhobeg': 0.5}, bounds=[(0., 1.)])

        if z_new.fun < z_min.fun:
            z_min = z_new
        print(z_range[0] + ((z_range[1]-z_range[0])*z_new.x), z_new.fun)
    return z_min

def calc_Pmin(KL_GP):
    PCE_DOE = srg.PCE_z_DOE(KL_GP)
    
    samples = KL_GP['gaussian'].sample(10000)

    res = np.zeros((samples.shape[1], len(PCE_DOE)))
    
    for i, PCE in enumerate(PCE_DOE):
        res[:, i] = PCE(*samples)
    min_idx = np.argmin(res, axis=1)
    print(min_idx.shape)

    is_min = np.zeros_like(res)
    is_min[np.arange(is_min.shape[0]), min_idx] = 1.
    

    Pmin = is_min.sum(axis=0)/is_min.shape[0]

    return Pmin
 
