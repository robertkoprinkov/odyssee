import numpy as np

from copy import deepcopy

from .KL_GP import KL_GP
from .SurrogateCoupledProblem import SurrogateCoupledProblem as SCP

class Optimize():

    def __init__(self, KL_GP, n_starting_points_z=20, eps_cv=0.01):
        self.KL_GP = deepcopy(KL_GP)
        
        self.n_starting_points_z = n_starting_points_z
        self.eps_cv = eps_cv
    
    """
        One optimization step. First calculates the PCE at the point z with the highest EI,
        which the GP approximation of the random field is then conditioned on. If necessary,
        the surrogates are then enriched at the point z with maximum Pmin among the points 
        with cv > eps_cv.

        @return Pmin after this optimization step
    """
    def optimization_step(self):
        z_next, f_z_next = self.KL_GP.z_next(n_starting_points=self.n_starting_points_z)
        print(z_next, f_z_next) 
        if np.abs(f_z_next) < 1e-6:
            print('Negligible improvement', z_next, f_z_next)
            return
        self.KL_GP.add_z(z_next)

        z_DoE, Pmin = self.KL_GP.Pmin()

        Pmin_order = np.argsort(-Pmin)

        z_DoE = z_DoE[Pmin_order]
        Pmin  = Pmin [Pmin_order]
        
        DoE_PCE = self.KL_GP.PCE_DoE
        
        for i, (z, p) in enumerate(zip(z_DoE, Pmin)):
            if p < (1./len(z_DoE)+1e-9):
                break
            print('z, p, cv', z, p, DoE_PCE[Pmin_order[i]].calc_cv()) 
            if DoE_PCE[Pmin_order[i]].calc_cv() > self.eps_cv:
                print('Enriching surrogates at z=', z)
                # enrich surrogates
                self.KL_GP.SCP.enrich_surrogates(z)
                
                # if this ends up taking long, this part can be optimized
                self.KL_GP = KL_GP(self.KL_GP.DoE_z, self.KL_GP.SCP, self.KL_GP.order, self.KL_GP.truncated)
                break
             
        return self.KL_GP.Pmin()
