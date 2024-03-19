import numpy as np

class ScalarCoupledProblem():

    def __init__(self, y1, y2, f_obj, lims_z, lims_y1, lims_y2):
        self.y1 = y1
        self.y2 = y2
        self.f_obj = f_obj

        self.lims_z = lims_z
        self.lims_y1 = lims_y1
        self.lims_y2 = lims_y2
        
        if len(lims_z.shape) == 1:
            self.lims_z = np.array([self.lims_z])
            self.dim_z = 1
        else:
            self.dim_z = lims_z.shape[0]

        if len(lims_y1.shape) == 1:
            self.lims_y1 = np.array([self.lims_y1])
            self.dim_y1 = 1
        else:
            self.dim_y1 = lims_y1.shape[0]

        if len(lims_y2.shape) == 1:
            self.lims_y2 = np.array([self.lims_y2])
            self.dim_y2 = 1
        else:
            self.dim_y2 = lims_y2.shape[0]
    """
        Solve the coupled problem with Gauß-Seidel iteration.
        @param z np.ndarray of dimension self.dim_z, or of shape (n_solves, self.dim_z), 
                 in which case the coupled problem is solved for n_solves values of z.
                 If self.dim_z is 1, a float will also be accepted.
        @param n_iter Maximum number of Gauß-Seidel iterations
        @param tol_abs Absolute convergence tolerance.

        @return converged, y Two np.ndarray, one of bool type showing whether convergence
                             was achieved, and the second giving the last value of y.
    """
    def solve(self, z, n_iter=1000, tol_abs=1e-6):
        if isinstance(z, float):
            assert(self.dim_z == 1)
            z = np.array([z])

        if len(z.shape) == 1:
            z = np.array([z])

        n_solves = z.shape[0]

        y = np.zeros((n_solves, self.dim_y1 + self.dim_y2))
        y_prev = y.copy()

        for n in range(n_iter):
            y_prev = y.copy()
            y[:, :self.dim_y1] = self.y1(z, y[:, self.dim_y1:])
            y[:, self.dim_y1:] = self.y2(z, y[:, :self.dim_y1])
            
            if np.all(np.abs(y_prev-y) < tol_abs): # converged
                break

        converged = np.all(np.abs(y_prev - y) < tol_abs, axis=1)

        return converged, y
