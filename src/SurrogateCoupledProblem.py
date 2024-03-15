import numpy as np
from .ScalarCoupledProblem import ScalarCoupledProblem as ScalarCoupledProblem
from .SurrogateFunction import Surrogate as Surrogate

from smt.sampling_methods import LHS

class SurrogateCoupledProblem(ScalarCoupledProblem):
    
    """
        Surrogate Coupled Problem

        @param f Function that takes parameters (z, y1, y2), of dimensions (n_evals, dim_z),
                 (n_evals, dim_y1) and (n_evals, dim_y2) respectively. z can alternatively
                 be of size (dim_z,), in which case each function evaluations are requested
                 for the same value of z
    """
    def __init__(self, scalar_coupled_problem, DoE_y1, DoE_y2):
        self.rng = np.random.default_rng()
        self.ScalarCP = scalar_coupled_problem
        
        self.lims_z = self.ScalarCP.lims_z
        self.lims_y1 = self.ScalarCP.lims_y1
        self.lims_y2 = self.ScalarCP.lims_y2

        self.dim_z = self.ScalarCP.dim_z
        self.dim_y1 = self.ScalarCP.dim_y1
        self.dim_y2 = self.ScalarCP.dim_y2

        if type(DoE_y1) is int:
            sampling_y1 = LHS(xlimits=np.concatenate((self.lims_z, self.lims_y2), axis=0), random_state=20)
            DoE_y1 = sampling_y1(DoE_y1)
        
        if type(DoE_y2) is int:
            sampling_y2 = LHS(xlimits=np.concatenate((self.lims_z, self.lims_y1), axis=0), random_state=20)
            DoE_y2 = sampling_y2(DoE_y2)
        
        self.y1 = Surrogate(self.ScalarCP.y1, DoE_y1[:, :self.dim_z], DoE_y1[:, self.dim_z:])
        self.y2 = Surrogate(self.ScalarCP.y2, DoE_y2[:, :self.dim_z], DoE_y2[:, self.dim_z:])
        self.f_obj = self.ScalarCP.f_obj

        # check sizes in unit tests
    """
        Solve the surrogate coupled problem with design/global variables z and random
        realizations from the Gaussian Process given by xi using Gauß-Seidel iteration.
        @param z np.ndarray of dimension self.dim_z, or of shape (n_solves, self.dim_z), 
                 in which case the coupled problem is solved for n_solves values of z.
                 If self.dim_z is 1, a float will also be accepted.
        @param xi (Optional) np.ndarray of dimension (n_solves, 2). This array determines
                  the drawn realizations from the Gaussian Processes, with the kth draw
                  being given by y_{i} = GP_{i, mean} + GP_{i, \\sigma}\\xi_{k, i}
                  If xi is not given, xi is drawn from the normal distribution.
        @param n_iter Maximum number of Gauß-Seidel iterations
        @param tol_abs Absolute convergence tolerance.

        @return converged, y Two np.ndarray, one of bool type showing whether convergence
                             was achieved, and the second giving the last value of y.
    """
    def solve(self, z, xi=None, n_iter=1000, tol_abs=1e-6):
        if type(z) == float or type(z) == np.float64: # include as optino everywhere
            assert(self.dim_z == 1)
            z = np.array([z])
        if len(z.shape) == 1:
            z = np.array([z])
        
        if xi is not None and len(xi.shape) == 1:
            xi = np.array([xi])

        if z.shape[0] == 1 and xi is not None:
            z = np.repeat(z, xi.shape[0], axis=0)
        n_solves = z.shape[0]
        
        if xi is None:
            xi = self.rng.normal(size=(n_solves, 2))
        assert(xi.shape[0] == z.shape[0])
        y = np.zeros((n_solves, self.dim_y1 + self.dim_y2))
        y_prev = y.copy()

        for n in range(n_iter):
            y_prev = y.copy()
            
            y[:, :self.dim_y1] = self.y1(z, y[:, self.dim_y1:], xi=xi[:, 0])
            y[:, self.dim_y1:] = self.y2(z, y[:, :self.dim_y1], xi=xi[:, 1])
            
            if np.all(np.abs(y_prev-y) < tol_abs): # converged
                break
        
        converged = np.all(np.abs(y_prev - y) < tol_abs, axis=1)
        return converged, y

    """
        Enrich surrogates at point z.
        @param z np.ndarray of size (self.dim_z) or (1, self.dim_z)
                 if self.dim_z is 1, a float will also be accepted
    """
    def enrich_surrogates(self, z):
        # enrich surrogates at this point
        converged, y = self.solve(z, xi=np.array([[0., 0.]]))
        
        if not np.all(converged):
            print('[GP enrichment] Solution to determine coupling variables diverged. ', z, y)

        self.y1.enrich(z, y[:, 1])
        self.y2.enrich(z, y[:, 0])
