import numpy as np

from smt.surrogate_models import KRG
from smt.sampling_methods import LHS

from scipy import stats
from scipy.optimize import minimize

from copy import deepcopy

from tqdm import tqdm

from .PCE import PCE

class KL_GP():

    def __init__(self, DoE_z, SCP, order, truncated=True):
        self.rng = np.random.default_rng()

        
        self.DoE_z = DoE_z.copy()
        if len(self.DoE_z.shape) == 1: # sequence of floats
            self.DoE_z = self.DoE_z.reshape(-1, 1)
        self.SCP = deepcopy(SCP)
        
        # used for finding starting points in the optimization for finding z_next
        self.sampling_z = LHS(xlimits=np.repeat([[0., 1.]], self.DoE_z.shape[1], axis=0))
         
        self.order = order
        self.truncated = truncated
        
        self.PCE_DoE = []
        
        self.A = []

        for z in DoE_z:
            self.PCE_DoE.append(PCE(z, SCP, order))
            self.A.append(self.PCE_DoE[-1].hermite_coefficients.copy())
        
        self.expansion = deepcopy(self.PCE_DoE[-1].expansion)
        self.A = np.array(self.A).reshape((len(DoE_z), self.expansion.size))
        
        self.phi_GP = [KRG(theta0=[1e-2]*np.ones(self.SCP.dim_z), print_prediction=False, print_global=False)]

        self._init_KL_GP()
    
    """
        Initialize KL_GP from the array self.A. Calculates eigenvalues phi and trains the interpolating GPs.
    """
    def _init_KL_GP(self):
        K = np.zeros((len(self.DoE_z), len(self.DoE_z)))

        for i in range(1, self.A.shape[1]):
            K += self.A[:, i:i+1] @ self.A[:, i:i+1].T

        self.eigval, self.phi = np.linalg.eig(K)

        self.eigval = self.eigval.real
        self.phi = self.phi.real
        
        # sort in order of decreasing eigenvalue
        phi_ordering = np.argsort(-self.eigval)
        
        self.eigval = self.eigval[phi_ordering]
        self.phi    = self.phi[phi_ordering]
        
        # train GPs interpolations and calculation truncation of eigenvectors
        self.phi_GP[0].set_training_values(self.DoE_z, self.A[:, 0])
        self.phi_GP[0].train()
        
        cumulative = np.cumsum(self.eigval/np.sum(self.eigval))
        self.phi_GP_energy = cumulative
        self.covered = cumulative > 1-1e-6

        if self.truncated:
            # better name => n_truncated_terms?
            self.truncation_index = np.arange(self.covered.shape[0])[self.covered].min()+1
        else:
            self.truncation_index = self.phi.shape[0]

        for coeff_i, phi_i in enumerate(self.phi):
            if coeff_i+2 > len(self.phi_GP):
                self.phi_GP.append(KRG(theta0=[1e-2]*np.ones(self.SCP.dim_z), print_prediction=False, print_global=False))
            GP = self.phi_GP[coeff_i+1]
            GP.set_training_values(self.DoE_z, phi_i)
            GP.train()

            if self.truncated and self.covered[coeff_i]:
                break
      
    """
        Get PCE expansion at one single point. 
    """
    def getPCE(self, z, eta=None):
        if eta is None:
            eta = self.rng.normal(size=(self.truncation_index+1))
        assert(eta.shape[0] == self.truncation_index+1)
        
        hermite_coefficients = np.zeros((self.A.shape[1], 1))

        hermite_coefficients[0, 0] = (self.phi_GP[0].predict_values(np.array([z])) + eta[0]*self.phi_GP[0].predict_variances(np.array([z]))).item()
        
        for k in range(self.truncation_index):
            for j in range(1, self.A.shape[1]):
                hermite_coefficients[j, 0] += ((self.A[:, j] @ self.phi[k])*(self.phi_GP[k+1].predict_values(np.array([z])) +
                                            eta[k+1]*self.phi_GP[k+1].predict_variances(np.array([z])))).item()
        return PCE(z, order=self.order, hermite_coefficients=hermite_coefficients, expansion=self.expansion)
    
    """
        Sample from KL GP at point z, using monte carlo sampling. xi and eta are the random variables that 
        determine the sample from the disciplinary solver GPs and the KL GP respectively.
        
        It is also possible to sample by calculating the mean and variance of the Gaussian distribution
        at z\\in Z with the self.getGaussianStatistics() function.
    """
    def sample(self, z, xi=None, eta=None, n_samples=10000):
        if isinstance(z, float) and self.SCP.dim_z == 1:
            z = np.array([z])
        if len(z.shape) == 1:
            z = np.repeat(np.array([z]), n_samples, axis=0)
        
        if xi is None:
            xi = self.rng.normal(size=(n_samples, 2))
        if eta is None:
            eta = self.rng.normal(size=(n_samples, self.truncation_index+1))

        assert (xi.shape[0]  == n_samples)
        assert (z.shape[0]   == n_samples)
        assert (eta.shape[0] == n_samples)

        mean = self.phi_GP[0].predict_values(z).flatten()
        var  = self.phi_GP[0].predict_variances(z).flatten()
        
        ret = np.zeros((n_samples, 1))
        
        ret[:, 0] = mean + eta[:, 0] * np.sqrt(var)
        
        for k in range(self.truncation_index): # only support one-dimensional output for now
            mean_gp = self.phi_GP[k+1].predict_values(z).flatten()
            sigma_gp  = eta[:, k] * np.sqrt(self.phi_GP[k+1].predict_variances(z).flatten())
            
            for j in range(1, self.A.shape[1]):
                mean = (self.A[:, j] @ self.phi[k] * self.expansion[j](xi[:, 0], xi[:, 1]))               * mean_gp
                sigma = (self.A[:, j] @ self.phi[k] * self.expansion[j](xi[:, 0], xi[:, 1]))              * sigma_gp
                
                ret[:, 0] += mean + sigma
        
        return ret

    """
        Returns mean and variance of Random Variable Y(z, \\xi_k, \\mu) for fixed \\xi_k. This RV is Gaussian,
        and its mean and variance describe it completely.

        @param z  np.ndarray of size (n_samples, self.SCP.dim_z) or (self.SCP.dim_z,), in which case it is assumed that
                  all samples are obtained at the same point z in space. If self.SCP.dim_z is 1, a float is also accepted.
        @param xi (Optional) np.ndarray of size (n_samples, 2). If not supplied, sampled from normal distribution.
        @param n_samples Number of samples.
        @param return_intermediate Returns all intermediate means and standard deviations if True.
    
        @return xi, mean, sigma and optionally mean_intermediate and sigma_intermediate, np.ndarrays of sizes respectively (n_samples, 2), (n_samples), (n_samples) (self.truncation_index+1, n_samples) and (self.truncation_index+1, n_samples)
        
    """
    def getGaussianStatistics(self, z, xi=None, n_samples=10000, return_intermediate=False):
        if isinstance(z, float) and self.SCP.dim_z == 1:
            z = np.array([z])
        if len(z.shape) == 1:
            z = np.repeat(np.array([z]), n_samples, axis=0)
        
        if xi is None:
            xi = self.rng.normal(size=(n_samples, 2))
            
        assert(xi.shape[0] == n_samples and z.shape[0] == n_samples)
        
        mean = self.phi_GP[0].predict_values(z).flatten()
        var  = self.phi_GP[0].predict_variances(z).flatten()
        
        mean_intermediate = [mean.copy()]
        var_intermediate = [var.copy()]
        for k in range(self.truncation_index): # only support one-dimensional output for now
            mean_gp = self.phi_GP[k+1].predict_values(z).flatten()
            var_gp  = self.phi_GP[k+1].predict_variances(z).flatten()
            
            for j in range(1, self.A.shape[1]):
                mean += (self.A[:, j] @ self.phi[k] * self.expansion[j](xi[:, 0], xi[:, 1]))             * mean_gp
                var  += np.power(self.A[:, j] @ self.phi[k] * self.expansion[j](xi[:, 0], xi[:, 1]), 2.) * var_gp

            if return_intermediate:
                mean_intermediate.append(mean.copy())
                var_intermediate.append(var.copy())
        sigma = np.sqrt(var)
        mean_intermediate = np.array(mean_intermediate)
        sigma_intermediate = np.sqrt(np.array(var_intermediate))

        if return_intermediate:
            return xi, mean, sigma, mean_intermediate, sigma_intermediate
        else:
            return xi, mean, sigma
    
    # does not work yet
    # z must be constant
    def getGaussianStatisticsVectorized(self, z, xi=None, n_samples=10000):
        if len(z.shape) == 1:
            z = np.array([z])

        if xi is None:
            xi = self.rng.normal(size=(n_samples, 2))
            
        assert(xi.shape[0] == n_samples and z.shape[0] == 1)
        assert(xi.shape[1] == 2 and z.shape[1] == self.SCP.dim_z)
        
        mean = []
        var = []
        
        for k in range(self.truncation_index):
            mean.append(self.phi_GP[k+1].predict_values(z).flatten())
            var.append(self.phi_GP[k+1].predict_variances(z).flatten())
        mean = np.array(mean)
        var = np.array(var)
        
        print(mean.shape, self.phi.shape, self.truncation_index, len(self.phi_GP))
        agg = self.A[:, 1:].T @ (self.phi @ np.diag(mean.flatten()))
        print(agg.shape)
        
        means_ret = self.phi_GP[0].predict_values(z).flatten() * np.ones((n_samples, 1))
        vars_ret = self.phi_GP[0].predict_variances(z).flatten() * np.ones((n_samples, 1))
        
        poly_evals = []
        for j in range(1, self.A.shape[1]):
            poly_eval = self.expansion[j](xi[:, 0], xi[:, 1])
            
            means_ret[:, 0] += np.sum(poly_eval.reshape(-1, 1) @ agg[j-1, :].reshape(1, -1), axis=1)
            vars_ret[:, 0] += np.sum(poly_eval.reshape(-1, 1) @ agg[j-1, :].reshape(1, -1), axis=1)
        
        sigma = np.sqrt(var)
        return xi, mean, sigma


    """
        Calculate EI for given point in space z.

        @param z np.ndarray of dimension(self.SCP.dim_z,) or, if self.SCP.dim_z is 1, a float.
                 (n_samples, self.SCP.dim_z) will also work, but in that case EI is calculated
                 for each (z_k, xi_k) pair, instead of for each (z_i, xi_k) pair.
        @param xi (Optional) np.ndarray of size (n_samples, 2)
        @param n_samples Number of samples to use in EI calculation
    """   
    def EI(self, z, xi=None, n_samples=10000):
        
        xi, mean, sigma = self.getGaussianStatistics(z, xi, n_samples)
        
        min_DoE = self.PCE_DoE[0](xi[:, 0], xi[:, 1])

        for i, z_ in enumerate(self.DoE_z[1:, :]):
            # too slow
            #_, mean_DoE, sigma_DoE = self.getGaussianStatistics(z_, xi)
            
            #assert np.max(np.abs(sigma_DoE)) < 1e-6, np.abs(np.max(sigma_DoE))
            
            mean_DoE = self.PCE_DoE[i](xi[:, 0], xi[:, 1])
            
            min_DoE = np.minimum(min_DoE, mean_DoE)
        
        EI_calc = (min_DoE - mean) * stats.norm.cdf((min_DoE - mean)/sigma)\
                  + sigma * stats.norm.pdf((min_DoE - mean)/sigma)
        
        if EI_calc.min() < 0.:
            print(z, np.max(sigma), sigma.min(), EI_calc.min(), mean, )
        return xi, np.mean(EI_calc)
        
    
    """
        Calculate EI with Monte Carlo sampling
    """
    def EI_sampling(self, z, n_samples=10000):
        pass


    """
        Add a given z to the DoE_z, and condition KL_GP on new z.
    """
    def add_z(self, z):
        if isinstance(z, float) and self.SCP.dim_z == 1:
            z = np.array([z])
        self.DoE_z = np.concatenate((self.DoE_z, np.array([z])), axis=0)
        self.PCE_DoE.append(PCE(z, self.SCP, self.order))
        
        self.A = np.concatenate((self.A, self.PCE_DoE[-1].hermite_coefficients.copy().T), axis=0)
        
        self._init_KL_GP()
    
    """
        Find the z with maximum EI

        @return argmax_{z} EI
    """
    def z_next(self, n_starting_points=20, log=True):
        z_range = self.SCP.lims_z
          
        def EI_(z, *args):
            # n inputs
            z = z_range[:, 0] + ((z_range[:, 1]-z_range[:, 0])*z)
            _, EI_calc = self.EI(z)
                
            return -EI_calc
        
        z_samples = self.sampling_z(n_starting_points)

        if log:
            print(z_samples[0])
        z_min = minimize(EI_, z_samples[0], method='COBYLA', options={'rhobeg': 0.5}, bounds=[(0., 1.)])
        
        if log:
            print(z_range[:, 0] + ((z_range[:, 1]-z_range[:, 0])*z_min.x), z_min.fun)
        
        for i in range(1, n_starting_points):
            z_new = minimize(EI_, z_samples[i], method='COBYLA', options={'rhobeg': 0.5}, bounds=[(0., 1.)])
            if log:
                print(z_range[:, 0] + ((z_range[:, 1]-z_range[:, 0])*z_new.x), z_new.fun)
            if z_new.fun < z_min.fun:
                z_min = z_new
        return z_range[:, 0] + ((z_range[:, 1]-z_range[:, 0])*z_min.x), z_min.fun
    
    """
        Calculates the probability of each z_i in the DoE_z of being the minimum point through Monte Carlo.

        @return DoE_z, Pmin_z
    """
    def Pmin(self, n_samples=10000):
        xi = self.rng.normal(size=(n_samples, 2))
        
        res = np.zeros((xi.shape[0], len(self.PCE_DoE)))
        
        for i, PCE in enumerate(self.PCE_DoE):
            res[:, i] = PCE(xi[:, 0], xi[:, 1])
        min_idx = np.argmin(res, axis=1)
        
        is_min = np.zeros_like(res)
        is_min[np.arange(is_min.shape[0]), min_idx] = 1.
    
        Pmin = is_min.sum(axis=0)/is_min.shape[0]

        return self.DoE_z, Pmin
