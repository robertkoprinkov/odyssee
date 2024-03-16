import numpy as np
import chaospy as cp

from copy import deepcopy

class PCE():
    
    """
        Initialize PCE in one of two ways: 

        - by propagating the uncertainty of a Surrogate Coupled Problem, 
          modeled by the variables xi, through the surrogate coupled 
          problem and through the objective function.
        - explicitly, by passing the used fourier coefficients and the used expansion.
        The former can be used to construct the KL-GP approximation of the
        random field of the objective function w.r.t. the design/optimisation variables,
        while the latter can be used to sample a PCE at any point of the KL-GP
        approximation.
        
        One of (SCP, order) or (fourier_coefficients, expansion) must be provided.
        
        @param z     At which the PCE is constructed
        @param SCP   SurrogateCoupledProblem
        @param order Order of the desired PCE expansion
        @param N     Number of samples to use when building the PCE expansion
        @param hermite_coefficients Coefficients of each term of the expansion. Typically used
                                    with the Hermite expansion.
        @param expansion            The used expansion, typically of hermite type.
    """
    def __init__(self, z, SCP=None, order=None, N=100, hermite_coefficients=None, expansion=None):
        self.rng = np.random.default_rng()
         
        self.SCP = SCP
        self.order = order

        if SCP is not None:
            self.expansion = cp.generate_expansion(order, cp.J(cp.Normal(0., 1.), cp.Normal(0., 1.)))
            
            xi_samples = self.rng.normal(size=(10*self.expansion.size, 2))
            
            converged, y_sol = SCP.solve(z, xi=xi_samples)
            
            if not np.all(converged): # filter to converged samples only
                print('Converged: %.2f percent' % (100*np.sum(converged)/converged.size), z)
                y_sol = y_sol[converged]
                xi_samples = xi_samples[converged, :]
            
            if (type(z) == float or type(z) == np.float64) and self.SCP.dim_z == 1:
                z = np.array([z])
            if len(z.shape) == 1:
                z = np.array([z])
            f_val = SCP.f_obj(z, y_sol[:self.expansion.size, :SCP.dim_y1], y_sol[:self.expansion.size, SCP.dim_y1:])
            
            self.PCE, self.hermite_coefficients = cp.fit_regression(self.expansion, xi_samples[:self.expansion.size, :].T, f_val, retall=1)
        
        else:
            self.expansion = deepcopy(expansion)
            self.hermite_coefficients = hermite_coefficients.copy()
            
            assert(len(self.expansion) == len(self.hermite_coefficients))
            
            self.PCE = self.hermite_coefficients[0]*self.expansion[0]
            
            for i in range(1, self.hermite_coefficients.shape[0]):
                self.PCE = self.PCE + self.hermite_coefficients[i] * self.expansion[i]
    
    def __call__(self, xi0, xi1):
        return self.PCE(xi0, xi1)

    """
        Sample from PCE

        @param N Number of desired samples.
    """
    def sample(self, N):
        xi_samples = self.rng.normal(size=(N, 2))

        return self.PCE(xi_samples[:, 0], xi_samples[:, 1])

    """
        Calculate cv
        @param N Number of samples to use when estimating the mean and the standard deviation.
    """
    def calc_cv(self, N=10000):
        samples = self.rng.normal(size=(N, 2))
        vals = self.PCE(samples[:, 0], samples[:, 1])
        
        mean_pce = np.mean(vals)
        sigma_pce = np.std(vals, ddof=1)

        if np.abs(mean_pce) < 1e-9:
            return sigma_pce
        else:
            # should we have abs here?
            return sigma_pce/np.abs(mean_pce)
