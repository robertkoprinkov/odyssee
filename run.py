import src.surrogate as srg
import src.vis as vis

import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

def y1(z, y2):
    return np.power(z, 2.) - np.cos(y2/2.)
def y2(z, y1):
    return z + y1
def f(z, y1, y2):
    return np.cos((y1 + np.exp(-y2))/np.pi) + z/20.

def solve(z):
     # solve MDA
    y = np.zeros(2)

    # Gauß iteration
    for i in range(100):
        y[0] = y1(z, y[1])
        y[1] = y2(z, y[0])
    return y

fig, ax = plt.subplots()
x = np.linspace(0., 20., 1000)
ax.plot(x, y1(-3., x))
ax.plot(y2(-3, x), x)

plt.savefig('img/analytic.png')

lims = np.array([[-5., 5.], [0., 20.]])
sampling = LHS(xlimits=lims, random_state=425)

# comment the next line to get unconvergent behavior
# this is caused by the fact that the chosen samples end up unlucky
# and as a result, the resulting graph is badly behaved and the map becomes non-contractive (!!)
xtest = sampling(500)

x1, srg1 = srg.get_surrogate(y1, sampling, 5)
x2, srg2 = srg.get_surrogate(y2, sampling, 4)
surrogates = [srg1, srg2]

print('Surrogates trained')
#solve_surrogate(surrogates, np.array([-0.06033744951679995, 0.4593774744332897]), -3.)
vis.plot_surrogates(*surrogates, lims[1], lims[1], -3., filename='img/surrogates.png')

PCE = srg.fit_PCE(surrogates, f, -3., 3, N=100)
vis.visualize_PCE(PCE, lim=(-1.5, 0), N=1000, filename='img/PCE.png')

KL_GP = srg.construct_KL_GP(surrogates, f, np.array([-4.3, -2.5, 0.5, 3.7]), 3, N=100)

print('KL_GP built')

z = np.linspace(-5., 5., 1000)
exact = np.zeros((2, z.shape[0]))
exact[0, :] = z

for i, z_ in enumerate(z):
    exact[1, i] = f(z_, *solve(z_))

srg.EI(KL_GP, -3.)

vis.visualize_KL_GP_eigenvectors(KL_GP, exact, (-5, 5.), filename='img/KL_GP_eigenvectors.png')
vis.visualize_KL_GP_distr(KL_GP, -3., lim=(-1.5, 0.), N=1000, filename='img/KL_GP_distr.png')
vis.visualize_KL_GP_distr(KL_GP, -3., lim=(-1.5, 0.), N=1000, filename='img/KL_GP_distr_copy.png')
vis.visualize_KL_GP_distr(KL_GP, -2.5, lim=(-1.5, 0.), N=1000, filename='img/KL_GP_distr_25.png')

