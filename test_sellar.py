import numpy as np
from src.ScalarCoupledProblem import ScalarCoupledProblem as ScalarCP
from src.SurrogateCoupledProblem import SurrogateCoupledProblem as SCP
from src.PCE import PCE
from src.KL_GP import KL_GP
from src.optimize import Optimize

from src.visualize import Plotter

from smt.sampling_methods import LHS

from tqdm import tqdm

import matplotlib.pyplot as plt

def y1(z, y2):
    return z[:, 0] + np.power(z[:, 1], 2.) + z[:, 2] - 0.2*y2.reshape(-1, 1)[:, 0]
def y2(z, y1):
    return np.sqrt(y1.reshape(-1, 1)[:, 0]) + z[:, 0] + z[:, 1]
def f(z, y1, y2):
    return (z[:, 0] + np.power(z[:, 2], 2) + np.exp(-y2.reshape(-1, 1)[:, 0]) + 10*np.cos(z[:, 1])).reshape(-1, 1)

problem = ScalarCP(y1, y2, f, np.array([[0., 10.], [-10., 10.], [0., 10.]]), np.array([1., 50.]), np.array([-1., 24.]))

sproblem = SCP(problem, 5, 5)

plotter = Plotter('img_sellar', sproblem)

DoE_z = LHS(xlimits=problem.lims_z, random_state=20)(20)
klgp = KL_GP(DoE_z, sproblem, 3)

optim = Optimize(klgp, n_starting_points_z=20)

for i in tqdm(range(10)):
    #fig, ax = plotter.plot_surrogates(optim.KL_GP, -3., 'img/%d_surrogate.png' % i)
    optim.KL_GP.add_z(np.array([0., 2.634, 0.]))
    print(optim.optimization_step())
    print(optim.KL_GP.SCP.y1.DoE_z)

    print('Truncation index', optim.KL_GP.truncation_index)
    #fig, ax = plotter.plot_surrogates(optim.KL_GP, optim.KL_GP.SCP.y1.DoE_z[-1], 'img/%d_surrogate_enriched.png' % i)

    #ax.scatter(optim.KL_GP.SCP.y2.f_DoE[-1], optim.KL_GP.SCP.y2.DoE_coupling[-1], color='green', marker='x')
    #ax.scatter(optim.KL_GP.SCP.y1.DoE_coupling[-1], optim.KL_GP.SCP.y1.f_DoE[-1], color='blue', marker='x')
    
    #ax.set_title('z=%.2f; cv=%.2f' % (optim.KL_GP.SCP.y1.DoE_z[-1], 0.2))#optim.KL_GP.PCE_DoE[optim.KL_GP.DoE_z == optim.KL_GP.SCP.y1.DoE_z[-1]].calc_cv()))

    #plt.figure(fig)
    #plt.savefig('img/%d_surrogate_enriched.png' % i)
