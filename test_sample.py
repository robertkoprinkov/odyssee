import numpy as np
from src.ScalarCoupledProblem import ScalarCoupledProblem as ScalarCP
from src.SurrogateCoupledProblem import SurrogateCoupledProblem as SCP
from src.PCE import PCE
from src.KL_GP import KL_GP
from src.optimize import Optimize

from src.visualize import Plotter

from tqdm import tqdm

import matplotlib.pyplot as plt

def y1(z, y2):
    return np.power(z, 2.) - np.cos(y2/2.)
def y2(z, y1):
    return z + y1
def f(z, y1, y2):
    return np.cos((y1 + np.exp(-y2))/np.pi) + z/20.

problem = ScalarCP(y1, y2, f, np.array([-5., 5.]), np.array([0., 20.]), np.array([0., 20.]))

print(problem.solve(-3.))

sproblem = SCP(problem, 5, 4)
print(sproblem.solve(-3., np.zeros((2, 2))))

pce = PCE(-3., sproblem, N=100, order=3)
print(pce.sample(10), pce.calc_cv())

klgp = KL_GP(np.array([-4.5, -2.3, 0.5, 3.5]), sproblem, 3)

plotter = Plotter('img', sproblem)
fig_EI, ax_EI = plotter.plot_EI(klgp)

klgp.getPCE(0.1)
print(klgp.getGaussianStatistics(-3.))
klgp.EI(2.)
print(klgp.Pmin())

#print(klgp.z_next())
optim = Optimize(klgp, n_starting_points_z=10)

for i in tqdm(range(10)):
    print(optim.optimization_step())
    ax_EI.axvline(optim.KL_GP.DoE_z[-1], linestyle='--')
    plt.figure(fig_EI)
    plt.savefig('img/%d_EI.png' % i)
    
    fig_EI, ax_EI = plotter.plot_EI(optim.KL_GP)

    fig, ax = plotter.plot_surrogates(optim.KL_GP, -3., 'img/%d_surrogate.png' % i)
    
    print(optim.KL_GP.SCP.y1.DoE_z)
    fig, ax = plotter.plot_surrogates(optim.KL_GP, optim.KL_GP.SCP.y1.DoE_z[-1], 'img/%d_surrogate_enriched.png' % i)

    ax.scatter(optim.KL_GP.SCP.y2.f_DoE[-1], optim.KL_GP.SCP.y2.DoE_coupling[-1], color='green', marker='x')
    ax.scatter(optim.KL_GP.SCP.y1.DoE_coupling[-1], optim.KL_GP.SCP.y1.f_DoE[-1], color='blue', marker='x')
    
    ax.set_title('z=%.2f; cv=%.2f' % (optim.KL_GP.SCP.y1.DoE_z[-1], 0.2))#optim.KL_GP.PCE_DoE[optim.KL_GP.DoE_z == optim.KL_GP.SCP.y1.DoE_z[-1]].calc_cv()))

    plt.figure(fig)
    plt.savefig('img/%d_surrogate_enriched.png' % i)
